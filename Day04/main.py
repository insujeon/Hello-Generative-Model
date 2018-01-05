import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from utils.visdom_utils import VisFunc

env_name = 'infoGAN'
vf = VisFunc(enval=env_name)

# Datasets
batch_size = 100
dataset = dset.MNIST('./dataset', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

#Models
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class FrontEnd(nn.Module):
    def __init__(self):
        super(FrontEnd, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1,64,4,2,1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024,7,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self,x):
        output = self.main(x)
        return output

class Dmodel(nn.Module):
    def __init__(self):
        super(Dmodel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1024,1,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        output=self.main(x).view(-1,1)
        return output

class Qmodel(nn.Module):
    def __init__(self):
        super(Qmodel,self).__init__()

        self.conv = nn.Conv2d(1024,128,1,bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(128,10,1)
        self.conv_mu = nn.Conv2d(128,2,1)
        self.conv_var = nn.Conv2d(128,2,1)

    def forward(self,x):
        y = self.conv(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var

class Gmodel(nn.Module):
    def __init__(self):
        super(Gmodel, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(74, 1024,1,1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 128,7,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,1,4,2,1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

FE=FrontEnd()
D=Dmodel()
Q=Qmodel()
G=Gmodel()

import ipdb; ipdb.set_trace(context=20)
for i in [FE, D, Q, G]:
    i.cuda()
    #initialize_weights(i)
    i.apply(weights_init)

# Define Loss
class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
                (x-mu).pow(2).div(var.mul(2.0)+1e-6)
        return logli.sum(1).mean().mul(-1)

criterionD = nn.BCELoss()
criterionQ_dis = nn.CrossEntropyLoss()
criterionQ_con = log_gaussian()

# Optimizers
optimD = optim.Adam([{'params':FE.parameters()},
                     {'params':D.parameters()}],
                    lr=0.0002, betas=(0.5, 0.99) )

optimG = optim.Adam([{'params':G.parameters()},
                     {'params':Q.parameters()}],
                    lr=0.001, betas=(0.5, 0.99) )

#fixed random variables for test
c0 = torch.linspace(-1,1,10).view(-1,1).repeat(10,0)
c1 = torch.stack((c0, torch.zeros(1).expand_as(c0)),1).cuda()
c2 = torch.stack((torch.zeros(1).expand_as(c0), c0),1).cuda()
one_hot = torch.eye(10).repeat(1,1,10).view(100,10).cuda()
fix_noise = torch.Tensor(100, 62).uniform_(-1, 1).cuda()


#random noises
def _noise_sample(dis_c, con_c, noise, bs):
    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0
    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
    return z, idx


for epoch in range(100):
    for num_iters, batch_data in enumerate(dataloader,0):

        # real part
        optimD.zero_grad()

        x, _ = batch_data
        real_x = Variable(x.cuda())
        label = Variable(torch.ones(batch_size).float().cuda(), requires_grad=False)

        fe_out1 = FE(real_x)
        probs_real = D(fe_out1)
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # fake part
        dis_c = Variable(torch.FloatTensor(batch_size,10).cuda())
        con_c = Variable(torch.FloatTensor(batch_size,2).cuda())
        noise = Variable(torch.FloatTensor(batch_size,62).cuda())
        z, idx = _noise_sample(dis_c,con_c,noise,batch_size)

        fake_x = G(z)
        fe_out2 = FE(fake_x.detach())
        probs_fake = D(fe_out2)
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        optimD.step()

        # G and Q part
        optimG.zero_grad()

        fe_out = FE(fake_x)
        probs_fake = D(fe_out)
        label.data.fill_(1.0)
        reconstruct_loss = criterionD(probs_fake, label)

        q_logits, q_mu, q_var = Q(fe_out)
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_logits, target)
        import ipdb; ipdb.set_trace(context=20)
        con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

        G_loss = reconstruct_loss + dis_loss + con_loss
        G_loss.backward()
        optimG.step()

        if num_iters % 100 == 0:
            print('Epoch:{0}, Iter:{1}, Dloss: {2}, Gloss: {3}, Preal: {4}, Pfake: {5}'.format(
                epoch, num_iters, D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(), probs_real.data.mean(), probs_fake.data.mean())
            )

            z = Variable(torch.cat([fix_noise, one_hot, c1], 1).view(-1, 74, 1, 1))
            x_save = G(z)
            save_image(x_save.data, './tmp/c1.png', nrow=10)
            title1 = '(C1) ' + str(epoch)+' eopch / '+str(num_iters) + ' iters'
            vf.imshow_multi(x_save.data.cpu(), nrow=10, title=title1,factor=1)

            z = Variable(torch.cat([fix_noise, one_hot, c2], 1).view(-1, 74, 1, 1))
            x_save = G(z)
            save_image(x_save.data, './tmp/c2.png', nrow=10)
            title2 = '(C2) ' + str(epoch)+' eopch / '+str(num_iters) + ' iters'
            vf.imshow_multi(x_save.data.cpu(), nrow=10, title=title2,factor=1)

