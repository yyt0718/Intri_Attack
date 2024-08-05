import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision.models as models
from utils import model_imgnet
from torch.utils.data import DataLoader
from utils import CustomDataset
from torch.utils.data import Dataset, DataLoader
import glob
from utils import loader_imgnet, model_imgnet, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uap_toep(model,modelname,loader,layers_indices,optimizer=None,scheduler=None,uap=None,eps=None, epochs=None,batch_size=None):
    '''
    :param model:model
    :param loder: dataloader
    :param optimizer: Adam or others
    :param scheduler: StepLR
    :param uap_initial:initial delta
    :param eps: maximum perturbation value (L-infinity) norm
    :param epoch:
    :return:
    '''
    device = torch.device("cuda:0")

    if eps is None:
        eps = 10
    if  epochs is None:
        epochs = 9000
    if uap is None:
        uap = torch.clamp(torch.rand(1, 3, 224, 224), -eps / 255, eps / 255).to(device)
    else:
        uap = torch.load(uap)
        uap = uap.to(device)
    uap.requires_grad_()
    if optimizer is None:
        optimizer = torch.optim.Adam([uap],lr=0.8)
    if scheduler is None:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.7)




    #获取layers_indices对应的vt
    def load_layers(indices):

        base_path = '/data/yyt/project1/data'
        loaded_tuples = []
        for index in indices:

            file_path = f"{base_path}/{modelname}/CONV{index}/CONV{index}_value_vec.pth"
            try:
                data_tuple = torch.load(file_path)
                data_tuple = (data_tuple[0].to(device), data_tuple[1].to(device))
                loaded_tuples.append(data_tuple)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                loaded_tuples.append(None)

        return loaded_tuples

    vt_s = load_layers(layers_indices)
    cos = nn.CosineSimilarity(dim=1)


    def loss_fn(delta, vt_s):
        inner_products = []
        total_sum = sum(item[0] for item in vt_s)
        # vt_s[i][0] / total_sum
        # 1 / len(delta)
        for k in range(batch_size):
            inner_product = 0  # inner product
            for i, layer_diff in enumerate(delta):
                #inner product
                weighted_inner_product = torch.abs(torch.matmul(layer_diff[k].view(-1).to(float), vt_s[i][1].squeeze(0).to(float)))
                inner_product += weighted_inner_product

            inner_products.append(inner_product)

        avg_inner_product = torch.mean(torch.stack(inner_products))
        loss = -avg_inner_product
        return loss, inner_products, avg_inner_product



    # register hooks
    def register_hooks(model, layers_indices):

        outputs = []
        hooks = []
        layers_indices = [layer - 1 for layer in layers_indices]
        vgg_model = model[1].module

        def hook_function(module, input, output):
            outputs.append(output)

        num_features = len(vgg_model.features)
        for layer in layers_indices:
            if layer == -1:
                #hook in Normalizer
                hook = model[0].register_forward_hook(hook_function)
            elif layer < num_features:
                #hook in features
                hook = vgg_model.features[layer].register_forward_hook(hook_function)
            elif layer == num_features:
                #hook in avgpool
                hook = vgg_model.avgpool.register_forward_hook(hook_function)
            else:
                #hook in classifier
                classifier_layer = layer - (num_features + 1)
                hook = vgg_model.classifier[classifier_layer].register_forward_hook(hook_function)
            hooks.append(hook)

        return outputs, hooks

    for epoch in range(epochs):
        # print('epoch %i/%i' % (epoch + 1, epochs))
        for i, data in enumerate(loader):
            x_prior = data.to(device)
            optimizer.zero_grad()
            # forward1
            outputs1, hooks1 = register_hooks(model, layers_indices)
            model(x_prior)
            # remove hook1
            for hook in hooks1:
                hook.remove()
            # forward2
            outputs2, hooks2 = register_hooks(model, layers_indices)
            model(torch.clamp(x_prior + uap,0,1))
            # remove hook2
            for hook in hooks2:
                hook.remove()
            # initial delta list
            delta = []
            # delta differences
            for j in range(len(layers_indices)):
                layers_diff = outputs2[j] - outputs1[j]
                delta.append(layers_diff)

            loss1 , _ , _ = loss_fn(delta,vt_s)

            loss = loss1
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            uap.data = torch.clamp(uap.data, -eps/255, eps/255)
            current_loss = loss.item()
            if (epoch + 1) % 50 == 0:
             print(f"Iteration {epoch}:loss {loss1.item()}")

    torch.save(uap, f'/data/yyt/project1/data/uaps/vgg19_norm10.pth')
    return uap.data ,loss

if __name__ == '__main__':
    model = model_imgnet('vgg19').eval()
    directory = '/data/yyt/project1/data/avgpix/mean'
    dataset = CustomDataset(directory)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # loader = loader_imgnet(dir_data, 48, batch_size, model_dimension, center_crop)

    modelname = 'vgg19'
    uap_toep(model,modelname ,data_loader, [0,2, 5, 7, 10, 12,14,16,19,21,23,25,28,30,32,34,38,41,44]  , optimizer=None, scheduler=None, uap="/data/yyt/project1/data/uaps/uap_init1.pth", eps=6, epochs=1000,
             batch_size=1)

#vgg16:[0,2,5,7,10,12,14,17,19,21,24,26,28,32,35,38]:93.4%
#vgg19:[0,2, 5, 7, 10, 12,14,16,19,21,23,25,28,30,32,34,38,41,44]:92%
#alexnet:[0,3,6,8,10,15,18,20]:94%
#googlenet:
#resnet: