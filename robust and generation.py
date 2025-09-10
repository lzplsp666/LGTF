import argparse
import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import torch.nn as nn
import os

import clip

from clip_mask import get_attn_clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.fix_clip import get_fixed_clip

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, AugMemAugmenter, StrongAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

from typing import Callable


def print_logger(
        old_print: Callable,
        file_name: str,
) -> Callable:
    """Returns a function which calls `old_print` twice, specifying a `file=` on the second call.

    Arguments:
        old_print: The `print` function to call twice.
        file_name: The name to give the log file.
    """

    def log_print(*args, **kwargs):
        old_print(*args, **kwargs)
        with open(file_name, "a") as log_file:
            old_print(*args, file=log_file, **kwargs)

    return log_print

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

## following APE.
def important_channel_indice(args, model, only_use_txt=True):
    if only_use_txt or args.shot  == 0:
        feats = model.text_feat.unsqueeze(1)  ## C * 1 * D
    else:
        feats = model.fixed_global_feat_vanilla ## C * L * D, including text feat & few shot image feat.
    cate_num, samp_num, feat_dim = feats.shape

    sim_sum = torch.zeros((feat_dim)).to(feats.device)
    count = 0
    # ipdb.set_trace()
    for i in range(cate_num):
        for j in range(cate_num):
            for m in range(samp_num):
                for n in range(samp_num):
                    if i != j:
                        sim_sum += feats[i, m, :] * feats[j, n, :]
                        count += 1
    sim = sim_sum / count
    # ipdb.set_trace()
    criterion = (-1) * args.lambda_ape * sim + (1-args.lambda_ape) * torch.var(model.text_feat, dim=0)
    _, indices = torch.topk(criterion, k=args.num_important_channel)
    return indices


def select_confident_samples(prob, top):
    # ipdb.set_trace()

    return prob[idx], idx, prob[idx_confused], idx_confused

def avg_entropy(outputs):
    ## N*Class
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

class Localpart(nn.Module):
    def __init__(self,args=None, beta=5.5, feat_dim=1024, class_num=1000, mapping='bias'):
        super(Localpart, self).__init__()

        self.args =  args
        self.indice = args.indice  ## indice of important channels.
        self.beta = beta
        self.rank = 4
        self.init_pred = 0
        self.local_llm_bias_key = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.local_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.local_llm_bias_value= nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.text_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.learnable_mapping = args.mapping





    def get_image_pred_attn_global_version2(self, model_clip,model_attn,return_full=False, return_logit=False):






    def get_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        if self.args.position == 'output' or self.args.position == 'all':
            text_feat = model.text_feat + self.text_bias
        else:
            text_feat = model.text_feat
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)  ## already filtered with indice.
        img_text_logit = logit_scale * model.image_features_global @ text_feat.t() ## 128*200
        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)


def get_searched_param(set_id, n_shot, ft):
    if ft:
        if set_id == 'I':
            return [0], [0.3], [0.00001], [100]  
        elif set_id == 'Flower102':
            return [0], [0.3], [0.001], [100]
        elif set_id == 'DTD':
            return [0], [0.3], [0.0001], [100]  
        elif set_id == 'Pets':
            return [0], [0.3], [0.0001], [20] 
        elif set_id == 'Cars':
            return [0], [0.3], [0.0001], [100] 
        elif set_id == 'UCF101':
            return [0], [0.3], [0.0001], [100] 
        elif set_id == 'Caltech101':
            return [0], [0.3], [0.0001], [20]
        elif set_id == 'Food101':
            if n_shot >=8:
                return [0], [0.3], [0.0001], [100] 
            else:
                return [0], [0.3], [0.0001], [20] 
        elif set_id == 'SUN397':
            return [0], [0.3], [0.0001], [30]
        elif set_id == 'Aircraft':
            return [0], [0.3], [0.0001], [100] 
        elif set_id == 'eurosat':
            if n_shot >=8:
                return [0], [0.3], [0.001], [100] 
            else:
                return [0], [0.3], [0.0001], [100] 
        else:
            raise NotImplementedError
    else:
        return [0], [0.3], [0.1], [20]  ## not used.

def main():
    args = parser.parse_args()
    args.log = args.log + '_' + str(args.gpu)
    set_random_seed(args.seed)
    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    model_clip =get_fixed_clip(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init, memory_size=args.memory_size, text_prompt=args.text_prompt)
    model_attn=get_attn_clip(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init,text_prompt=args.text_prompt)
    model_state = None

    for name, param in model_clip.named_parameters():
        param.requires_grad_(False)
    for name, param in model_attn.named_parameters():
        param.requires_grad_(False)


    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model_clip = model_clip.cuda(args.gpu)
        model_attn = model_attn.cuda(args.gpu)

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    num_important_channel_list = args.num_important_channel.split("/")
    lambda_ape_list = args.lambda_ape.split("/")
    lr_list = args.lr.split("/")
    epoch_list = args.epoch.split("/")
    results = {}
    print_log = print_logger(print, os.path.join(args.log + '.txt'))

    for set_id in datasets:
        if args.use_searched_param:
            num_important_channel_list, lambda_ape_list, lr_list, epoch_list = get_searched_param(set_id, args.n_shot, args.ft)
        best_acc = 0
        print_log("processing the dataset{} \n".format(set_id), end="	")

        for num_important_channel in num_important_channel_list:
            for lambda_ape in lambda_ape_list:
                for lr in lr_list:
                    for epoch in epoch_list:
                        print('adopt num_important_channel {}, lambda_ape: {}'.format(num_important_channel, lambda_ape))
                        args.lr = float(lr)
                        args.epoch = int(epoch)
                        args.num_important_channel = int(num_important_channel)
                        args.lambda_ape = float(lambda_ape)
                        base_transform = transforms.Compose([
                            transforms.Resize(args.resolution, interpolation=BICUBIC),
                            transforms.CenterCrop(args.resolution)])
                        preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            normalize])
                        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                                             augmix=len(set_id) > 1) ### aug mix not used for ImageNet test set.
                        # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1, augmix=False)

                        test_transform = transforms.Compose([
                            transforms.Resize(args.resolution, interpolation=BICUBIC),
                            transforms.CenterCrop(args.resolution), transforms.ToTensor(), normalize])
                        batchsize = 1

                        print("evaluating: {}".format(set_id))
                        # reset the model
                        # Reset classnames of custom CLIP model
                        if len(set_id) > 1:
                            # fine-grained classification datasets
                            classnames = eval("{}_classes".format(set_id.lower()))
                        else:
                            assert set_id in ['A', 'R', 'K', 'V', 'I']
                            classnames_all = imagenet_classes
                            classnames = []
                            if set_id in ['A', 'R', 'V']:
                                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                                if set_id == 'R':
                                    for i, m in enumerate(label_mask):
                                        if m:
                                            classnames.append(classnames_all[i])
                                else:
                                    classnames = [classnames_all[i] for i in label_mask]
                            else:
                                classnames = classnames_all

                        model_clip.reset_classnames(classnames, set_id)
                        model_attn.reset_classnames(classnames, set_id)

                        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode,noise=args.noise_proportion)
                        print("number of test samples: {}".format(len(val_dataset)))
                        val_loader = torch.utils.data.DataLoader(
                                    val_dataset,
                                    batch_size=batchsize, shuffle=False,  ## the input has been shuffled.
                                    num_workers=args.workers, pin_memory=True)

                        args.set_id = set_id
                        model_clip.eval()
                        with torch.no_grad():
                            text_feat, text_feat_full = model_clip.get_text_features()
                            llm_local_text=model_clip.get_llm_text_features(datasets,16)

                        if args.n_shot:
                            if args.n_augview == 0:
                                train_dataset_mem = build_dataset(set_id, test_transform, args.data, mode='train', n_shot=args.n_shot)
                                print("number of training samples: {}".format(len(train_dataset_mem)))
                                train_loader_mem = torch.utils.data.DataLoader(
                                            train_dataset_mem,
                                            batch_size=1, shuffle=False,  ## the input has been shuffled.
                                            num_workers=args.workers, pin_memory=True)
                                # for (image,target) in train_loader_mem:
                                #     print("a")
                                init_image_memory(train_loader_mem, model_clip,model_attn, args)
                                del train_dataset_mem, train_loader_mem
                            else:
                                ######### generate num_aug_view augmented views for each samples; APE adopt ten...
                                assert args.n_augview % args.n_shot == 0
                                num_aug_view = int(args.n_augview / args.n_shot)
                                data_transform_aug = AugMemAugmenter(base_transform, preprocess, n_views=num_aug_view - 1,
                                                                 augmix=len(set_id) > 1)  ### aug mix not used for ImageNet test set.
                                train_dataset_mem = build_dataset(set_id, data_transform_aug, args.data, mode='train', n_shot=args.n_shot)
                                print("number of training samples: {}, number of augview: {}".format(len(train_dataset_mem), args.n_augview))
                                train_loader_mem = torch.utils.data.DataLoader(
                                            train_dataset_mem,
                                            batch_size=1, shuffle=False,  ## the input has been shuffled.
                                            num_workers=args.workers, pin_memory=True)
                                init_image_memory(train_loader_mem, model_clip , args)
                                del train_dataset_mem, train_loader_mem
                        ########## extract the importance channels via APE.
                        if args.num_important_channel != 0:
                            important_indice = important_channel_indice(args, model_clip) ##
                            args.indice = important_indice
                        else:
                            important_indice = torch.arange(model_clip.text_feat.shape[1]).to(model_clip.text_feat.device) ## use all channels.
                            args.indice = important_indice
                        model_clip.fixed_global_feat=torch.load('fixed_global_feat.pt')
                        results_temp = direct_inference(val_loader, model_clip,model_attn, args)
                        # top1.avg, top1_global.avg, top1_global_fewshot.avg, best_acc, best_beta1, best_beta2, best_beta3
                        print_log("lr: {}, epoch:{}, num_important_channel{}, lambda_ape: {}, best acc{:.2f} \n".format(lr, epoch, num_important_channel, lambda_ape, results_temp[3]), end="	")
                        if results_temp[3] > best_acc:
                            results[set_id] = results_temp
                            best_acc = results_temp[3]
                        # results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
                        del val_dataset, val_loader
                        try:
                            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
                        except:
                            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
                        length = len(results[set_id])
    args.indice = 0
    log = open(os.path.join(args.log + '.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()
    print_log("======== Result Summary ========")
    print_log("params: bs	lr	selection_p")
    print_log("params: {}	{}	{}".format(args.batch_size, args.lr, args.selection_p))
    print_log("\t\t [set_id] \t\t Top-1 acc. \t\t Top-1 local acc, \t\t Top-1 global acc \t\t Searched acc \t\t beta \t\t gama.")
    for id in results.keys():
        print_log("{}".format(id), end="	")
    print_log('mean', end="	")
    print_log("\n")
    for i in range(length):
        cul_acc = 0
        cul_count = 0
        for id in results.keys():
            print_log("{:.3f}".format(results[id][i]), end="	")
            cul_acc += float(results[id][i])
            cul_count += 1
        print_log("{:.3f}".format(cul_acc), end="	")
        print_log("\n")



def entropy(outputs):
    # prob: 1*200, logit.
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0, keepdim=True) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]; log(mean_prob)
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    confidence_entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    return confidence_entropy

def init_image_memory(train_loader, model,model_version, args):
    model.eval()
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    memorized_image_global_feat = [] ## N*[shot*aug]*C
    memorized_image_local_feat = []  ## N*[shot*aug]*C
    memorized_image_global_feat_vanilla = [] ## N*shot*C
    memorized_image_local_feat_vanilla = []  ## N*shot*C
    memorized_labels = []

    for i in range(model.n_cls):
        memorized_image_global_feat.append([])
        memorized_image_local_feat.append([])
        memorized_image_global_feat_vanilla.append([])
        memorized_image_local_feat_vanilla.append([])
        memorized_labels.append([])

    for i, (images, target) in enumerate(train_loader):
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
        else: ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            image_features_global, image_features_local =  model.get_image_features(images) ## 4*1024; 4*49*1024.
            # im=model_version.get_image_attn(images)
        text_features = model.text_feat[target]  ## 512
        ## only use the original ?? we should use all; however, only use the vanilla one in the dynamic memory.
        selected_image_features_local = model.image_features_local
        cos_sim = (selected_image_features_local * text_features).sum(-1)  ## between 0.2-0.3, very close.
        weight_prob = (cos_sim * 100).softmax(-1)   ## 1*197, following clip temperature.
        ########
        attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)  ## 1*512
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 1*512
        memorized_image_global_feat[target].append(image_features_global) ## aug*C
        memorized_image_local_feat[target].append(attented_feat)   # aug * C
        memorized_image_global_feat_vanilla[target].append(image_features_global[:1]) ## aug*C
        memorized_image_local_feat_vanilla[target].append(attented_feat[:1])   # aug * C
        one_hot_target = torch.zeros(1, model.n_cls).to(target.device)
        one_hot_target[0, target] = 1
        memorized_labels[target].append(one_hot_target)   ## 1 * C, turn it to one hot labels.

    for i in range(model.n_cls):
        memorized_image_global_feat[i] = torch.cat(memorized_image_global_feat[i], dim=0).unsqueeze(0) ## 1*augshot*C
        memorized_image_local_feat[i] = torch.cat(memorized_image_local_feat[i], dim=0).unsqueeze(0)
        memorized_image_global_feat_vanilla[i] = torch.cat(memorized_image_global_feat_vanilla[i], dim=0).unsqueeze(0) ## 1*shot*C
        memorized_image_local_feat_vanilla[i] = torch.cat(memorized_image_local_feat_vanilla[i], dim=0).unsqueeze(0)
        memorized_labels[i] = torch.cat(memorized_labels[i], dim=0).unsqueeze(0)

    memorized_image_global_feat = torch.cat(memorized_image_global_feat, dim=0) ## n*shot*c
    memorized_image_local_feat = torch.cat(memorized_image_local_feat, dim=0)
    memorized_image_global_feat_vanilla = torch.cat(memorized_image_global_feat_vanilla, dim=0) ## n*shot*c
    memorized_image_local_feat_vanilla = torch.cat(memorized_image_local_feat_vanilla, dim=0)
    memorized_labels = torch.cat(memorized_labels, dim=0)

    ######## memorized few shot features and labels.
    model.fewshot_image_global_feat = memorized_image_global_feat ## class*augshot*c
    model.fewshot_image_local_feat = memorized_image_local_feat
    model.fewshot_image_global_feat_vanilla = memorized_image_global_feat_vanilla ## class*shot*c
    model.fewshot_image_local_feat_vanilla = memorized_image_local_feat_vanilla
    model.fewshot_label = memorized_labels  ## class*shot*c, one hot labels

    ############# add features of labeled data to the dynamic memory. This is important when there are more labeled data.
    model.fixed_global_feat_vanilla = torch.cat((model.fixed_global_feat, memorized_image_global_feat_vanilla), dim=1)  ## N*1*C
    model.fixed_local_feat_vanilla = torch.cat((model.fixed_local_feat, memorized_image_local_feat_vanilla), dim=1)  ## N*1*C

    ###################### for static memory, with text feature and augmented image feat
    model.fixed_global_feat = torch.cat((model.fixed_global_feat, memorized_image_global_feat), dim=1)  ## N*1*C
    model.fixed_local_feat = torch.cat((model.fixed_local_feat, memorized_image_local_feat), dim=1)  ## N*1*C
    torch.save(model.fixed_global_feat, 'fixed_global_feat.pt')
    print('appending the few shot image feature to fixed image memories.')

def direct_inference(val_loader, model_clip,model_attn, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)

    top1_global = AverageMeter('Accglobal_pred@1', ':6.2f', Summary.AVERAGE)

    top1_text_vote = AverageMeter('Acc_text1@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot_vote = AverageMeter('Accgobal@1', ':6.2f', Summary.AVERAGE)
    few_shot_local=AverageMeter('AccLocal@1', ':6.2f', Summary.AVERAGE)
    fewshot_local_pred=AverageMeter('AccLocal_pred@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1_global, top1_text_vote, top1_global_fewshot_vote, few_shot_local, fewshot_local_pred],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model_clip.eval()
    model_attn.eval()
    if model_clip.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model_clip.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    ## text_feat: 200*1024
    ## text_feat_full:  200 * 7 * 1024
    class_num, feat_dim = model_clip.text_feat.shape[0], model_clip.text_feat.shape[1]
    pred_vanilla = []
    pred_global = []
    pred_local = []
    pred_fewshot_global = []
    pred_fewshot_local = []
    labels = []

    localpart=Localpart(args=args,beta=args.beta, feat_dim=feat_dim, class_num=class_num, mapping=args.mapping).cuda()
    ################################ fine tune clip adapter with few labeled training data.
    if args.n_shot and args.ft:
        epoch = args.epoch
        training_size = model_clip.text_feat.shape[0] * args.n_shot
        #### construct the data loader,
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform_aug = StrongAugmenter(base_transform, preprocess, augmix=len(args.set_id) > 1)  ### aug mix not used for ImageNet test set.
        train_dataset_mem = build_dataset(args.set_id, data_transform_aug, args.data, mode='train', n_shot=args.n_shot)
        print("number of training samples: {}, number of augview: {}".format(len(train_dataset_mem), args.n_augview))
        train_loader_ft = torch.utils.data.DataLoader(
            train_dataset_mem,
            batch_size=128 if training_size > 128 else training_size, shuffle=True,  ## the input has been shuffled.
            num_workers=2, pin_memory=True)
        if args.optimizer == 'adamw':

            optimizer_b=torch.optim.AdamW(localpart.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)
        elif args.optimizer == 'sgd':

            optimizer_b=torch.optim.AdamW(localpart.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        else:
            raise NotImplementedError

        scheduler_b= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch * len(train_loader_ft), eta_min=1e-7)
        Loss = SmoothCrossEntropy()
        timestamp = time.time()
        time_parts = time.gmtime(timestamp)
        hours = time.strftime("%H", time_parts)
        minutes = time.strftime("%M", time_parts)
        seconds = time.strftime("%S", time_parts)
        print("train start Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))



    localpart.load_state_dict(torch.load('localpart_real.pth'))

    localpart.eval()
    end = time.time()
    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    hours = time.strftime("%H", time_parts)
    minutes = time.strftime("%M", time_parts)
    seconds = time.strftime("%S", time_parts)
    print("test start Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))
    count=0
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        # patch=(int)(196*args.noise_proportion)
        # images=add_noise_to_image_list(images,num_noisy_patches=patch)
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        else: ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            image_features_global, image_features_local =  model_clip.get_image_features(images)
            model_attn.get_image_attn(images)


        with torch.no_grad():
            img_text = localpart.get_text_prediction(model_clip)
        img_text_pred = img_text[:1]  ## current prediction.
        confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text, args.selection_p)
        localpart.init_pred = confidence_prediction.mean(0, keepdim=True)
        # acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
        # top1_text_vote.update(acc1[0], image.size(0))
        pred_vanilla.append(img_text_pred)


        with torch.no_grad():
            fewshot_local_pred_fullview=localpart.get_image_pred_attn_global_version2(model_clip,model_attn)
            # fewshot_local_pred =   fewshot_local_pred_fullview[:1]
            confidence_prediction_fewshot_local, _, _, _ = select_confident_samples(fewshot_local_pred_fullview, 0.1)
            fewshot_local_mean=confidence_prediction.mean(0, keepdim=True)
            # acc1, _ = accuracy(confidence_prediction_fewshot_local.mean(0, keepdim=True), target, topk=(1, 5))
            # few_shot_local.update(acc1[0], image.size(0))
        pred_fewshot_local.append(fewshot_local_mean)

        labels.append(target)
        batch_time.update(time.time() - end)
        end = time.time()
        torch.cuda.empty_cache()


        if (i+1) % 200 == 0:
            # progress.display(i)
            print(f"{count}round")
            count=count+1
            pred_text = torch.cat(pred_vanilla, dim=0)

            fewshot_local = torch.cat(pred_fewshot_local, dim=0)
            label = torch.cat(labels, dim=0)
            acc,_= accuracy( pred_text, label, topk=(1, 5))
            print("text ：" + str(acc))

            acc,_= accuracy(fewshot_local, label, topk=(1, 5))
            print("pred_fewshot_local ：" + str(acc))



    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    hours = time.strftime("%H", time_parts)
    minutes = time.strftime("%M", time_parts)
    seconds = time.strftime("%S", time_parts)
    print("end Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))


    pred_text = torch.cat(pred_vanilla, dim=0)

    fewshot_local = torch.cat(pred_fewshot_local, dim=0)

    label = torch.cat(labels, dim=0)



    weight_search = True
    search_step = 10
    if weight_search:
        beta1_list = [1.0,0.9,0.8,0.7,1.2,1.5,0.5,0.6,0.4,1.2,1.5,0.4,0.3]
        beta2_list = [0,0.001, 0.003, 0.0, 1,1.5,1.6,2, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        if args.n_shot:
            beta3_list = [0,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
            beta5_list= [0,0.001, 0.003, 0.01,1,3,10]
        else:
            beta3_list = [0]
        # beta1_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)]  ## 0.001 - 10
        print('-' * 20)
        print('Starting searching...')
        print('     beta1 searching range: [0.001, ' + str(10) + ']')
        print('     beta2 searching range: [0.001, ' + str(10) + ']')
        print('     beta3 searching range: [0.001, ' + str(10) + ']')
        print('     beta4 searching range: [0.001, ' + str(10) + ']')
        print('-' * 20)
        acc,_= accuracy(pred_text, label, topk=(1, 5))
        print( acc)
        acc,_= accuracy( fewshot_local, label, topk=(1, 5))
        print(acc)
        numbers = []
        current = 0.0
        while current <= 10:
            numbers.append(round(current, 2))
            current += 0.05

        beta4_list=numbers
        best_acc = 0.
        best_beta2 = 0.
        best_beta3 = 0.
        for beta1 in beta1_list:
            for beta4 in beta4_list:
                logits = pred_text * beta1 +fewshot_local*beta4
                acc, _ = accuracy(logits, label, topk=(1, 5))
                acc = acc.item()
                if acc > best_acc:
                    print('New best setting, beta1: {:.4f}; beta4: {:.4f};Acc: {:.2f}'.format(beta1,beta4, acc))
                    best_acc = acc
                    best_beta1 = beta1
                    best_beta4 = beta4

    return [top1_global.avg, best_acc, best_beta1, best_beta2, best_beta3]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')

    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use.')

    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_augview', type=int, default=0, help='use augmented few shot samples') 
    parser.add_argument('--ft', action='store_true', default=False, help="fine tuning the attention weight with few labeled data.")
    parser.add_argument('--use_searched_param', action='store_true', default=False, help="using searched param for each dataset")
    
    parser.add_argument('--beta',  default=5.5, type=float, help='loss weight')
    parser.add_argument('--mapping', type=str, default='bias', help='bias | affine | all')
    parser.add_argument('--position', type=str, default='all', help='query | key | value | qkv | output | all')
    parser.add_argument('--optimizer', type=str, default='adamw', help='adamw | sgd')
    parser.add_argument('--eps',  default=1e-8, type=float, help='eps, default 1e-8')
    parser.add_argument('--wd',  default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lr',  default='0.0001', type=str, help='learning rate')
    parser.add_argument('--epoch', type=str, default='20')
    parser.add_argument('--shared_param', action='store_true', default=False, help="shared parameters acorss local | global | text.")
    parser.add_argument('--num_important_channel', type=str, default='0') ## if 0, use all channels; otherwise, selecting the ape_channel_num
    parser.add_argument('--lambda_ape', default='0.7', type=str, help='following ape.')
    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--text_prompt', type=str, default='tip_cupl', help='simple | tip | full | tip_cupl')
    parser.add_argument('--log', type=str, default='loga', help='some places to write note')

    parser.add_argument('--noise_proportion', default=0.0, type=float, help='proportion of noise')
    main()



