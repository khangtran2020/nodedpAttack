import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from Trim.appeardict import AppearDict
from loguru import logger
from Utils.utils import timeit

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score


def update_clean(model, optimizer, objective, batch):
    optimizer.zero_grad()
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[0].srcdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(mfgs, inputs)
    # print(f"Size of predictions: {predictions.size()}, Size of labels: {labels.size()}")
    loss = objective(predictions, labels)
    loss.backward()
    optimizer.step()
    return labels, predictions, loss
    

def update_nodedp(args, model, optimizer, objective, batch, g, clip_grad,
                  clip_node, ns, trim_rule, history, step, device):
    optimizer.zero_grad()
    dst_node, subgraphs = batch
    dst_node = list(dst_node)
    if trim_rule == 'impact':
        appear_dict = AppearDict(roots=dst_node, subgraph=subgraphs, graph=g, clip_node=clip_node, rule=trim_rule,
                                 num_layer=args.n_layers, debug=args.debug, step=step, device=device, model=model)
        info = appear_dict.trim()
        history['% subgraph'].append(info['num_subgraphs_trimmed'] / info['num_subgraphs'])
        total = 0
        for root in info['trimmed_subgraphs']:
            total += info[root]['num_node_trimmed'] / info[root]['num_node_org']
        history['% node avg'].append(total / (info['num_subgraphs_trimmed'] + 1e-12))
        blocks = appear_dict.joint_blocks()
    else:
        with torch.no_grad():
            appear_dict = AppearDict(roots=dst_node, subgraph=subgraphs, graph=g, clip_node=clip_node, rule=trim_rule,
                             num_layer=args.n_layers, debug=args.debug, step=step, device=device, model=None)
            info = appear_dict.trim()
            history['% subgraph'].append(info['num_subgraphs_trimmed'] / info['num_subgraphs'])
            total = 0
            for root in info['trimmed_subgraphs']:
                total += info[root]['num_node_trimmed'] / info[root]['num_node_org']
            history['% node avg'].append(total / (info['num_subgraphs_trimmed'] + 1e-12))
            blocks = appear_dict.joint_blocks()
    model.zero_grad()
    inputs = blocks[0].srcdata["feat"]
    labels = blocks[-1].dstdata["label"]
    predictions = model(blocks, inputs)
    losses = objective(predictions, labels)
    running_loss = torch.mean(losses).item()
    num_data = predictions.size(dim=0)

    saved_var = dict()
    for tensor_name, tensor in model.named_parameters():
        saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

    for pos, j in enumerate(losses):
        j.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)
        model.zero_grad()

    for tensor_name, tensor in model.named_parameters():
        if tensor.grad is not None:
            saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, ns * clip_grad * clip_node).to(device))
            tensor.grad = saved_var[tensor_name] / num_data

    optimizer.step()
    return labels, predictions, running_loss


def eval_clean(model, objective, batch):
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[0].srcdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(mfgs, inputs)
    loss = objective(predictions, labels)
    return labels, predictions.argmax(1), loss


def train_fn(dataloader, model, criterion, optimizer, device, metric, scheduler):
    model.to(device)
    model.train()
    train_loss = 0
    num_data = 0.0
    for bi, d in enumerate(dataloader):
        target, pred, loss = update_clean(model=model, optimizer=optimizer, objective=criterion, batch=d)
        if scheduler is not None:
            scheduler.step()
        metric.update(pred, target)
        num_data += pred.size(dim=0)
        train_loss += loss.item()*pred.size(dim=0)
    performance = metric.compute()
    metric.reset()
    return train_loss / num_data, performance


def train_nodedp(args, dataloader, model, criterion, optimizer, device, scheduler, g, clip_grad, clip_node, ns,
                 trim_rule, history, step, metric):
    model.to(device)
    g.to(device)
    model.train()
    train_loss = 0
    batch = next(iter(dataloader))
    target, pred, loss = update_nodedp(args=args, model=model, optimizer=optimizer, objective=criterion,
                                       batch=batch, g=g, clip_grad=clip_grad, clip_node=clip_node, ns=ns,
                                       trim_rule=trim_rule, history=history, step=step, device=device)
    train_loss += loss
    if scheduler is not None:
        scheduler.step()

    performace = metric(pred, target)
    metric.reset()

    return train_loss, performace


def eval_fn(data_loader, model, criterion, metric, device):
    model.to(device)
    loss_eval = 0
    model.eval()
    num_point = 0
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            target, pred, loss = eval_clean(model=model, objective=criterion, batch=d)
            num_pt = pred.size(dim=0)
            loss_eval += loss.item() * num_pt
            num_point += num_pt
            metric.update(pred, target)
        performance = metric.compute()
        metric.reset()
    return loss_eval / num_point, performance


def get_norm_grad(model):
    total_l2_norm = 0
    for p in model.named_parameters():
        total_l2_norm += p[1].grad.detach().norm(p=2) ** 2
    return np.sqrt(total_l2_norm)


def performace_eval(args, y_true, y_pred):
    if args.performance_metric == 'acc':
        return accuracy_score(y_true=y_true, y_pred=y_pred)
    elif args.performance_metric == 'f1':
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    elif args.performance_metric == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif args.performance_metric == 'pre':
        return precision_score(y_true=y_true, y_pred=y_pred)


def update_dp(model, optimizer, objective, batch, clip, ns):
    optimizer.zero_grad()
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[0].srcdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(mfgs, inputs)
    losses = objective(predictions, labels)
    running_loss = torch.mean(losses).item()
    num_data = predictions.size(dim=0)
    # print(losses)

    saved_var = dict()
    for tensor_name, tensor in model.named_parameters():
        saved_var[tensor_name] = torch.zeros_like(tensor)

    for pos, j in enumerate(losses):
        j.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)
        model.zero_grad()

    for tensor_name, tensor in model.named_parameters():
        if tensor.grad is not None:
            saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, ns * clip))
            tensor.grad = saved_var[tensor_name]  # / num_data

    optimizer.step()
    return labels, predictions.argmax(1), running_loss


def update_mlp_clean(model, optimizer, objective, loader, metrics):
    train_loss = 0
    num_data = 0
    for batch in loader:
        optimizer.zero_grad()
        input_nodes, output_nodes, mfgs = batch
        inputs = mfgs[-1].dstdata["feat"]
        labels = mfgs[-1].dstdata["label"]
        predictions = model(inputs)
        loss = objective(predictions, labels)
        loss.backward()
        optimizer.step()
        metrics.update(predictions, labels)
        num_data += predictions.size(dim=0)
        train_loss += loss.item() * predictions.size(dim=0)
    perf = metrics.compute()
    metrics.reset()
    return perf, train_loss / num_data

def update_mlp_dpsgd(model:torch.nn.Module, optimizer, objective, loader, clip_grad, ns, metrics, device):
    batch = next(iter(loader))
    optimizer.zero_grad()
    model.zero_grad()
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[-1].dstdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(inputs)
    loss = objective(predictions, labels)
    metrics.update(predictions, labels)
    train_loss = torch.mean(loss).item()
    num_data = predictions.size(dim = 0)
    
    saved_var = {}
    for tensor_name, tensor in model.named_parameters():
        saved_var[tensor_name] = torch.zeros_like(tensor).to(device)

    for pos, j in enumerate(loss):
        j.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                new_grad = tensor.grad
                saved_var[tensor_name].add_(new_grad)
        model.zero_grad()

    for tensor_name, tensor in model.named_parameters():
        if tensor.grad is not None:
            saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, ns*clip_grad).to(device))
            tensor.grad = saved_var[tensor_name] / num_data

    optimizer.step()
    perf = metrics.compute()
    metrics.reset()
    return perf, train_loss / num_data


def train_mlp(loader, model, criter, optimizer, device, metrics, mode='clean', clip=None, ns=None):
    
    model.to(device)
    model.train()

    if mode == 'clean':
        tr_acc, tr_loss = update_mlp_clean(model=model, optimizer=optimizer, objective=criter, loader=loader, metrics=metrics)
    else:
        tr_acc, tr_loss = update_mlp_dpsgd(model=model, optimizer=optimizer, objective=criter, loader=loader, clip_grad=clip, ns=ns, metrics=metrics, device=device)

    return tr_acc, tr_loss

def eval_mlp(loader, model, criter, metrics, device):
    model.to(device)
    loss_eval = 0
    model.eval()
    num_point = 0
    with torch.no_grad():
        for bi, d in enumerate(loader):
            input_nodes, output_nodes, mfgs = d
            inputs = mfgs[-1].dstdata["feat"]
            labels = mfgs[-1].dstdata["label"]
            predictions = model(inputs)
            loss = criter(predictions, labels)
            num_pt = predictions.size(dim=0)
            loss_eval += loss.item() * num_pt
            num_point += num_pt
            metrics.update(predictions, labels)
        performance = metrics.compute()
        metrics.reset()
    return performance, loss_eval / num_point
