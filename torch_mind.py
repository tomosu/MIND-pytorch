import torch
import math


class MIND(torch.nn.Module):

    def __init__(self, non_local_region_size =9, patch_size =7, neighbor_size =3, gaussian_patch_sigma =3.0):
        super(MIND, self).__init__()
        self.nl_size =non_local_region_size
        self.p_size =patch_size
        self.n_size =neighbor_size
        self.sigma2 =gaussian_patch_sigma *gaussian_patch_sigma


        # calc shifted images in non local region
        self.image_shifter =torch.nn.Conv2d(in_channels =1, out_channels =self.nl_size *self.nl_size,
                                            kernel_size =(self.nl_size, self.nl_size),
                                            stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2),
                                            dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            t =torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i%self.nl_size, i//self.nl_size] =1
            self.image_shifter.weight.data[i] =t


        # patch summation
        self.summation_patcher =torch.nn.Conv2d(in_channels =self.nl_size*self.nl_size, out_channels =self.nl_size*self.nl_size,
                                              kernel_size =(self.p_size, self.p_size),
                                              stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                              dilation=1, groups=self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            # gaussian kernel
            t =torch.zeros((1, self.p_size, self.p_size))
            cx =(self.p_size-1)//2
            cy =(self.p_size-1)//2
            for j in range(self.p_size *self.p_size):
                x=j%self.p_size
                y=j//self.p_size
                d2 =torch.norm( torch.tensor([x-cx, y-cy]).float(), 2)
                t[0, x, y] =math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] =t


        # neighbor images
        self.neighbors =torch.nn.Conv2d(in_channels =1, out_channels =self.n_size*self.n_size,
                                        kernel_size =(self.n_size, self.n_size),
                                        stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2),
                                        dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t =torch.zeros((1, self.n_size, self.n_size))
            t[0, i%self.n_size, i//self.n_size] =1
            self.neighbors.weight.data[i] =t


        # neighbor patcher
        self.neighbor_summation_patcher =torch.nn.Conv2d(in_channels =self.n_size*self.n_size, out_channels =self.n_size*self.n_size,
                                               kernel_size =(self.p_size, self.p_size),
                                               stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                               dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t =torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] =t



    def forward(self, orig):
        assert(len(orig.shape) ==4)
        assert(orig.shape[1] ==1)

        # get original image channel stack
        orig_stack =torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size)], dim=1)

        # get shifted images
        shifted =self.image_shifter(orig)

        # get image diff
        diff_images =shifted -orig_stack

        # diff's L2 norm
        Dx_alpha =self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images =self.neighbor_summation_patcher( self.neighbors(orig) )
        Vx =neighbor_images.var(dim =1).unsqueeze(dim =1)

        # output mind
        nume =torch.exp(-Dx_alpha /(Vx +1e-8))
        denomi =nume.sum(dim =1).unsqueeze(dim =1)
        mind =nume /denomi
        return mind


class MINDLoss(torch.nn.Module):

    def __init__(self, non_local_region_size =9, patch_size =7, neighbor_size =3, gaussian_patch_sigma =3.0):
        super(MINDLoss, self).__init__()
        self.nl_size =non_local_region_size
        self.MIND =MIND(non_local_region_size =non_local_region_size,
                        patch_size =patch_size,
                        neighbor_size =neighbor_size,
                        gaussian_patch_sigma =gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind =self.MIND(input)
        tar_mind =self.MIND(target)
        mind_diff =in_mind -tar_mind
        l1 =torch.norm( mind_diff, 1)
        return l1/(input.shape[2] *input.shape[3] *self.nl_size *self.nl_size)


if __name__ =="__main__":
    mind =MIND()
    orig =torch.ones(4,1,128,128)
    mind(orig)
