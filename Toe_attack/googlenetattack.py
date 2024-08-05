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
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uap_toep(model,modelname,loader,layers_indices1,layers_indices2,optimizer=None,scheduler=None,uap=None,eps=None, epochs=None,batch_size=None):
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
    def load_layers(indices,name):

        base_path = '/home/yyt/Project1/data'
        loaded_tuples = []
        for index in indices:
            file_path = os.path.join(base_path, modelname, name, f'convbatch_{index}.pth')

            try:
                data_tuple = torch.load(file_path)
                data_tuple = (data_tuple[0].to(device), data_tuple[1].to(device))
                loaded_tuples.append(data_tuple)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                loaded_tuples.append(None)

        return loaded_tuples

    layers_indices3=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


    vt_s1 = load_layers(layers_indices2,name='all_b1')
    vt_s2 = load_layers(layers_indices1, name='all_b2')
    vt_s3 = load_layers(layers_indices3, name='all_b3')
    vt_s4 = load_layers(layers_indices2, name='all_b4')




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
                # cos(layer_diff[k].view(-1).unsqueeze(0), vt_s[i][1]))
                #内积
                weighted_cos_sim =torch.abs(torch.matmul(layer_diff[k].view(-1).to(float), vt_s[i][1].squeeze(0).to(float)))
                cos_sim += weighted_cos_sim  # 累加到当前batch元素的总相似度

            cos_sims.append(cos_sim)

        avg_cos_sim = torch.mean(torch.stack(cos_sims))
        loss = -avg_cos_sim
        return loss, cos_sims, avg_cos_sim

    def register_hooks1(model):
        inputs = []
        hooks = []

        def get_input_activation():
            def hook(model, input, output):
                inputs.append(input[0])

            return hook



        inception_modules = [model[1].module.inception3a, model[1].module.inception3b, model[1].module.inception4a,
                             model[1].module.inception4b, model[1].module.inception4c, model[1].module.inception4d,
                             model[1].module.inception4e, model[1].module.inception5a, model[1].module.inception5b]

        for inception_module in inception_modules:
            hook = inception_module.branch1.conv.register_forward_hook(get_input_activation())
            hooks.append(hook)



        return inputs, hooks

    def register_hooks2(model):
        inputs = []
        hooks = []

        def get_input_activation():
            def hook(model, input, output):
                inputs.append(input[0])

            return hook

        # 注册钩子并将它们添加到hooks列表
        hook = model[1].module.conv1.conv.register_forward_hook(get_input_activation())
        hooks.append(hook)
        hook = model[1].module.conv2.conv.register_forward_hook(get_input_activation())
        hooks.append(hook)
        hook = model[1].module.conv3.conv.register_forward_hook(get_input_activation())
        hooks.append(hook)

        inception_modules = [model[1].module.inception3a, model[1].module.inception3b, model[1].module.inception4a,
                             model[1].module.inception4b, model[1].module.inception4c, model[1].module.inception4d,
                             model[1].module.inception4e, model[1].module.inception5a, model[1].module.inception5b]

        for inception_module in inception_modules:
            hook = inception_module.branch2[0].conv.register_forward_hook(get_input_activation())
            hooks.append(hook)
            hook = inception_module.branch2[1].conv.register_forward_hook(get_input_activation())
            hooks.append(hook)


        hook = model[1].module.fc.register_forward_hook(get_input_activation())
        hooks.append(hook)

        return inputs, hooks

    def register_hooks3(model):
        inputs = []
        hooks = []

        def get_input_activation():
            def hook(model, input, output):
                inputs.append(input[0])

            return hook


        inception_modules = [model[1].module.inception3a, model[1].module.inception3b, model[1].module.inception4a,
                             model[1].module.inception4b, model[1].module.inception4c, model[1].module.inception4d,
                             model[1].module.inception4e, model[1].module.inception5a, model[1].module.inception5b]

        for inception_module in inception_modules:
            hook = inception_module.branch3[0].conv.register_forward_hook(get_input_activation())
            hooks.append(hook)
            hook = inception_module.branch3[1].conv.register_forward_hook(get_input_activation())
            hooks.append(hook)


        return inputs, hooks

    def register_hooks4(model):
        inputs = []
        hooks = []

        def get_input_activation():
            def hook(model, input, output):
                inputs.append(input[0])

            return hook

        # 注册钩子并将它们添加到hooks列表


        inception_modules = [model[1].module.inception3a, model[1].module.inception3b, model[1].module.inception4a,
                             model[1].module.inception4b, model[1].module.inception4c, model[1].module.inception4d,
                             model[1].module.inception4e, model[1].module.inception5a, model[1].module.inception5b]

        for inception_module in inception_modules:
            hook = inception_module.branch4[1].conv.register_forward_hook(get_input_activation())
            hooks.append(hook)



        return inputs, hooks

    # def register_hooks2(model):
    #     outputs = []
    #     hooks = []
    #
    #     def get_output_activation():
    #         def hook(model, input, output):
    #             outputs.append(output)  # 使用detach()来避免保存不必要的计算图
    #
    #         return hook
    #
    #     # GoogLeNet中的Inception模块
    #     inception_modules = [
    #         model[1].module.inception3a, model[1].module.inception3b,
    #         model[1].module.inception4a, model[1].module.inception4b,
    #         model[1].module.inception4c, model[1].module.inception4d,
    #         model[1].module.inception4e, model[1].module.inception5a,
    #         model[1].module.inception5b
    #     ]
    #
    #     # 在每个Inception模块上注册钩子
    #     for module in inception_modules:
    #         hook = module.register_forward_hook(get_output_activation())
    #         hooks.append(hook)
    #
    #     return outputs, hooks

    for epoch in range(epochs):
        print('epoch %i/%i' % (epoch + 1, epochs))

        for i, data in enumerate(loader):
            x_prior = data.to(device)
            optimizer.zero_grad()
            # branch1
            outputs1, hooks1 = register_hooks1(model)
            model(x_prior)
            # remove hook1
            for hook in hooks1:
                hook.remove()
            # forward2
            outputs2, hooks2 = register_hooks1(model)
            model(torch.clamp(x_prior + uap, 0, 1))
            # remove hook2
            for hook in hooks2:
                hook.remove()

            # initial delta list
            delta1 = []
            # delta differences
            for j in range(len(layers_indices2)):
                layers_diff1 = outputs2[j] - outputs1[j]
                delta1.append(layers_diff1)





          #branch2

            outputs3, hooks3 = register_hooks2(model)
            model(x_prior)
            # remove hook1
            for hook in hooks3:
                hook.remove()
            # forward2
            outputs4, hooks4 = register_hooks2(model)
            model(torch.clamp(x_prior + uap, 0, 1))
            # remove hook2
            for hook in hooks4:
                hook.remove()

            # initial delta list
            delta2 = []
            # delta differences
            for j in range(len(layers_indices1)):
                layers_diff2 = outputs4[j] - outputs3[j]
                delta2.append(layers_diff2)



            # branch3
            outputs5, hooks5 = register_hooks3(model)
            model(x_prior)
            # remove hook1
            for hook in hooks5:
                hook.remove()
            # forward2
            outputs6, hooks6 = register_hooks3(model)
            model(torch.clamp(x_prior + uap, 0, 1))
            # remove hook2
            for hook in hooks6:
                hook.remove()

            # initial delta list
            delta3 = []
            # delta differences
            for j in range(len(layers_indices3)):
                layers_diff3 = outputs6[j] - outputs5[j]
                delta3.append(layers_diff3)



            #4


            outputs7, hooks7 = register_hooks4(model)
            model(x_prior)
            # remove hook1
            for hook in hooks7:
                hook.remove()
            # forward2
            outputs8, hooks8 = register_hooks4(model)
            model(torch.clamp(x_prior + uap, 0, 1))
            # remove hook2
            for hook in hooks8:
                hook.remove()

            # initial delta list
            delta4 = []
            # delta differences
            for j in range(len(layers_indices2)):
                layers_diff4 = outputs2[j] - outputs1[j]
                delta4.append(layers_diff4)











            loss1 , cos_sims , avg_cos_sim = loss_fn(delta1,vt_s1)
            loss2, cos_sims2, avg_cos_sim2 = loss_fn(delta2, vt_s2)
            loss3, cos_sims3, avg_cos_sim4 = loss_fn(delta3, vt_s3)
            loss4, cos_sims3, avg_cos_sim4 = loss_fn(delta4, vt_s4)


            loss = 5/10*loss2+ 2/10*loss1+3/10*loss3+0.5/10*loss4


            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            uap.data = torch.clamp(uap.data, -eps/255, eps/255)
            current_loss = loss.item()
            print(f"Iteration {epoch} ,loss2 {loss2.item()} , Loss1 {loss1.item()}")





    torch.save(uap, f'/home/yyt/Project1/data/uaps/googlenet/googlenetbr1234.pth')
    return uap.data ,loss

if __name__ == '__main__':
    model = model_imgnet('googlenet').eval()
    directory = '/home/yyt/Project1/data/avgpix/mean'
    dataset = CustomDataset(directory)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    modelname = 'googlenet'
    layers_indices1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    layers_indices2 = [3,4,5,6,7,8,9,10,11]


    uap_toep(model,modelname ,data_loader,layers_indices1,layers_indices2,optimizer=None, scheduler=None, uap="/home/yyt/Project1/data/uaps/uap_init1.pth", eps=10, epochs=500,
             batch_size=1)

    #layers_indices1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]