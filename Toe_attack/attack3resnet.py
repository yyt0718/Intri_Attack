import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import torchvision.models as models
from utils import model_imgnet
from torch.utils.data import DataLoader
from utils import CustomDataset
from torch.utils.data import Dataset, DataLoader
import glob
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uap_toep(model,modelname,loader,layers_indices,layers_indices2=None,optimizer=None,scheduler=None,uap=None,eps=None, epochs=None,batch_size=None):
    '''
    :param model:model
    :param loder: dataloader
    :param optimizer: Adam or others
    :param scheduler: StepLR
    :param uap_initial:initial delta
    :param xi:
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
        optimizer = torch.optim.Adam([uap],lr=0.5)
    if scheduler is None:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.7)
        # scheduler = CosineAnnealingLR(optimizer, T_max=1500, eta_min=0.001)




    #获取layers_indices对应的vt
    def load_layers(indices):

        base_path = '/home/yyt/Project1/data'
        loaded_tuples = []

        for index in indices:
            # file_path = f"{base_path}/{modelname}/all_152convs/conv{index}.pth"
            # file_path = f"{base_path}/{modelname}/convbatch152/convbarch_{index}.pth"
            file_path = f"{base_path}/{modelname}/block50/Block{index}.pth"
            try:
                data_tuple = torch.load(file_path)
                data_tuple = (data_tuple[0].to(device), data_tuple[1].to(device))
                loaded_tuples.append(data_tuple)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                loaded_tuples.append(None)

        return loaded_tuples

    if layers_indices2 is None:
        vt_s = load_layers(layers_indices)
    else:
        vt_s = load_layers(layers_indices)
        vt_s = [vt_s[i] for i in layers_indices2]

    cos = nn.CosineSimilarity(dim=1)

    def loss_Acv(outputs3):
        for k in range(batch_size):
            act=0
            acts = []
            for i, output in enumerate(outputs3):
                act += (torch.log(torch.norm(output[k],2)))
            acts.append(act)
        avg_cos_sim = sum(torch.stack(acts))
        loss2 = -avg_cos_sim
        return loss2

    def loss_fn(delta, vt_s):
        cos_sims = []
        total_sum = sum(item[0] for item in vt_s)
        # vt_s[i][0] / total_sum
        # 1 / len(delta)
        for k in range(batch_size):
            cos_sim = 0  # 在处理每个batch元素时重置cos_sim
            for i, layer_diff in enumerate(delta):
                # 加权余弦相似度最高60%fooling rate
                # weighted_cos_sim =  (vt_s[i][0] )*torch.abs(
                #     cos(layer_diff[k].view(-1).unsqueeze(0), vt_s[i][1]))
                #内积
                weighted_cos_sim =torch.abs(torch.matmul(layer_diff[k].view(-1).to(float), vt_s[i][1].squeeze(0).to(float)))

                cos_sim += weighted_cos_sim  # 累加到当前batch元素的总相似度

            cos_sims.append(cos_sim)

        avg_cos_sim = torch.mean(torch.stack(cos_sims))
        loss = -avg_cos_sim
        return loss, cos_sims, avg_cos_sim



    # def register_hooks(model):
    #     outputs = []
    #     hooks = []
    #
    #     def hook_fn(module, input, output):
    #         outputs.append(input[0])
    #
    #     for name, module in model.named_modules():
    #         if "conv" in name and isinstance(module, nn.Conv2d) and "downsample" not in name:
    #             hooks.append(module.register_forward_hook(hook_fn))
    #         elif name.endswith('fc'):
    #             hooks.append(module.register_forward_hook(hook_fn))
    #
    #     return outputs, hooks
    #


    def register_hooks2(model):
        outputs = []
        hooks = []

        def hook_fn(module, input, output):
            outputs.append(input[0])

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "downsample" not in name:
                if 'conv1' in name and 'layer' in name:  # 检查是否为瓶颈单元的开始
                    hooks.append(module.register_forward_hook(hook_fn))

        return outputs, hooks


    for epoch in range(epochs):
        print('epoch %i/%i' % (epoch + 1, epochs))

        for i, data in enumerate(loader):
            x_prior = data.to(device)
            optimizer.zero_grad()

            outputs1, hooks1 = register_hooks2(model)
            model(x_prior)
            # remove hook1
            for hook in hooks1:
                hook.remove()
            # forward2
            outputs2, hooks2 = register_hooks2(model)
            model(torch.clamp(x_prior + uap, 0, 1))
            # remove hook2
            for hook in hooks2:
                hook.remove()

            outputs3,hooks3=register_hooks2(model)
            model(torch.clamp(x_prior + uap, 0, 1))
            for hook in hooks3:
                hook.remove()


            # initial delta list
            delta = []
            # delta differences
            for j in range(len(layers_indices)):
                layers_diff = outputs2[j] - outputs1[j]
                delta.append(layers_diff)

            if layers_indices2 is not None:
                delta = [delta[i] for i in layers_indices2]
            # if layers_indices2 is not None:
            #     outputs2 = [outputs2[i] for i in layers_indices2]

            loss1 , cos_sims , avg_cos_sim = loss_fn(delta,vt_s)

            loss2 = loss_Acv(outputs3)
            # loss3 = loss_Acv(delta)
            loss =10/100*loss1+10/10*loss2

            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            uap.data = torch.clamp(uap.data, -eps/255, eps/255)
            current_loss = loss.item()
            print(f"Iteration {epoch}:loss1 {loss1.item()},loss2{loss2.item()} , Loss {loss.item()}")

    torch.save(uap, f'/home/yyt/Project1/data/uaps/resnet152/rresnnet.pth')
    return uap.data ,loss

if __name__ == '__main__':
    model = model_imgnet('resnet152').eval()
    directory = '/home/yyt/Project1/data/avgpix/mean'
    dataset = CustomDataset(directory)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    modelname = 'resnet152'
    layers_indices =  list(range(1,51))
    layers_indices2 = [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    # layers_indices2 = [0]  + [151
    # layers_indices2 = [11,12,13,14,15,16,17]

    uap_toep(model,modelname ,data_loader,layers_indices,layers_indices2,optimizer=None, scheduler=None, uap=None, eps=10, epochs=800,
             batch_size=1)

    #
    # layers_indices2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
 #
 # layers_indices2 = [1,3,5,7,9,11,12,13,14,15,16,17,19,21,23,26,27,28,29,30,34,35,37,38,40,41,42,43,44,45,47,49]
# layers_indices2 = layers_indices2 = [1,3,5,7,9,11,12,35,37,38,40,41,42,43,44,45,47,49] most now.
# layers_indices2 = layers_indices2 =  [1,3,5,9,11,12,13,14,16,18,19,20,22,23,25,27,29,30,32,33,35,36,37,38,39,41,43,45,46,47,49]


#1-20 , 10/100 64.18
#6:1-20,15/100  64.09%
#12: 1-20,7/100 64.10
#15:1-20, 11 62