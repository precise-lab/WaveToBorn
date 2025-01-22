import torch
import numpy as np
import time

class WaveFoward(torch.nn.Module):
    def __init__(self, nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = 1.5, abc = 1.0):
        super().__init__()
        times = torch.arange(nt)*dt
        self.source_intentisty = torch.sin(2*np.pi*fc*times)*torch.exp( -(times-t0)**2/(2*sgma**2))
        thetas = 2*np.pi*np.arange(nm)/nm
        meas_inds_x =(NX/2 + sR*np.cos(thetas)).astype(int)
        meas_inds_y =(NX/2 + sR*np.sin(thetas)).astype(int)
        self.meas_inds = NX*meas_inds_x + meas_inds_y
        self.trans_inds = self.meas_inds[::sr]

        self.nr = len(self.trans_inds)

        self.dx = dx
        self.nx = nx
        self.NX = NX
        
        self.nt = nt
        self.dt = dt
        self.bg = bg
        
        self.abc = abc
        self.abc_pd = NX//2 - sR - 1
        self.abc_pad = (self.abc_pd, self.abc_pd, self.abc_pd, self.abc_pd)
        self.pd =(self.NX- self.nx)//2 - self.abc_pd
        self.pad = (self.pd, self.pd, self.pd, self.pd)


    def forward(self, x):
        x = torch.nn.functional.pad(x, self.pad,    mode='constant', value=self.bg)
        x = torch.nn.functional.pad(x, self.abc_pad,    mode='constant', value=1e-6)
        

        out = []
        p0 = torch.zeros((x.shape[0], len(self.trans_inds), x.shape[2], x.shape[3]), device = x.get_device())
        p1 = torch.zeros((x.shape[0], len(self.trans_inds), x.shape[2], x.shape[3]), device = x.get_device())
        eye = torch.eye(len(self.trans_inds), device = x.get_device())
        

        for j in range(self.nt):
            print(j/self.nt)
            nab_p1_x = (-torch.roll(p1, 2, -2) + 16*torch.roll(p1, 1, -2)  - 30*p1 + 16*torch.roll(p1, -1, -2) - torch.roll(p1, -2, -2))/12
            nab_p1_y = (-torch.roll(p1, 2, -1) + 16*torch.roll(p1, 1, -1)  - 30*p1 + 16*torch.roll(p1, -1, -1) - torch.roll(p1, -2, -1))/12
            p = 2*p1  - p0 + (self.dt*x/self.dx)**2*(nab_p1_x + nab_p1_y)
            p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])
            p[:,:,self.trans_inds] = p[:,:,self.trans_inds] + (self.bg*self.dt/self.dx)**2*self.source_intentisty[j]*eye[None,:,:]
            out.append( p[:,:, self.meas_inds])
            p = p.reshape(p1.shape)

            p0 = p1
            p1 = p
        return torch.stack(out, dim = 3)
            
class SEWaveFoward(torch.nn.Module):
    def __init__(self, nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = 1.5, abc = 1.0):
        super().__init__()
        times = torch.arange(nt)*dt
        self.source_intentisty = torch.sin(2*np.pi*fc*times)*torch.exp( -(times-t0)**2/(2*sgma**2))
        thetas = 2*np.pi*np.arange(nm)/nm
        meas_inds_x =(NX/2 + sR*np.cos(thetas)).astype(int)
        meas_inds_y =(NX/2 + sR*np.sin(thetas)).astype(int)
        self.meas_inds = NX*meas_inds_x + meas_inds_y
        self.trans_inds = self.meas_inds[::sr]

        self.nr = len(self.trans_inds)

        self.dx = dx
        self.nx = nx
        self.NX = NX
        
        self.nt = nt
        self.dt = dt
        self.bg = bg
        
        self.abc = abc
        self.abc_pd = NX//2 - sR - 1
        self.abc_pad = (self.abc_pd, self.abc_pd, self.abc_pd, self.abc_pd)
        self.pd =(self.NX- self.nx)//2 - self.abc_pd
        self.pad = (self.pd, self.pd, self.pd, self.pd)


    def forward(self, x, se):
        x = torch.nn.functional.pad(x, self.pad,    mode='constant', value=self.bg)
        x = torch.nn.functional.pad(x, self.abc_pad,    mode='constant', value=1e-6)
        

        out = []
        p0 = torch.zeros_like(x)
        p1 = torch.zeros_like(x)
        

        for j in range(self.nt):
            nab_p1_x = (-torch.roll(p1, 2, -2) + 16*torch.roll(p1, 1, -2)  - 30*p1 + 16*torch.roll(p1, -1, -2) - torch.roll(p1, -2, -2))/12
            nab_p1_y = (-torch.roll(p1, 2, -1) + 16*torch.roll(p1, 1, -1)  - 30*p1 + 16*torch.roll(p1, -1, -1) - torch.roll(p1, -2, -1))/12
            p = 2*p1 - p0 + (self.dt*x/self.dx)**2*(nab_p1_x + nab_p1_y)
            p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])
            p[:,:,self.trans_inds] = p[:,:,self.trans_inds] + (self.bg*self.dt/self.dx)**2*self.source_intentisty[j]*se[None,None,:]
            out.append( p[:,:, self.meas_inds])
            p = p.reshape(p1.shape)

            p0 = p1
            p1 = p
        return torch.stack(out, dim = 3)

class HelperFoward(torch.nn.Module):
    def __init__(self, nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = 1.5, abc = 1.0):
        super().__init__()
        times = torch.arange(nt)*dt
        self.source_intentisty = torch.sin(2*np.pi*fc*times)*torch.exp( -(times-t0)**2/(2*sgma**2))
        thetas = 2*np.pi*np.arange(nm)/nm
        meas_inds_x =(NX/2 + sR*np.cos(thetas)).astype(int)
        meas_inds_y =(NX/2 + sR*np.sin(thetas)).astype(int)
        self.meas_inds = NX*meas_inds_x + meas_inds_y
        self.trans_inds = self.meas_inds[::sr]

        self.nr = len(self.trans_inds)

        self.dx = dx
        self.nx = nx
        self.NX = NX
        
        self.nt = nt
        self.dt = dt
        self.bg = bg
        
        self.abc = abc
        self.abc_pd = NX//2 - sR - 1
        self.abc_pad = (self.abc_pd, self.abc_pd, self.abc_pd, self.abc_pd)
        self.pd =(self.NX- self.nx)//2 - self.abc_pd
        self.pad = (self.pd, self.pd, self.pd, self.pd)


    def forward(self, x):
        x = torch.nn.functional.pad(x, self.pad,    mode='constant', value=self.bg)
        x = torch.nn.functional.pad(x, self.abc_pad,    mode='constant', value=1e-6)
        

        out = torch.zeros(x.shape[0], len(self.trans_inds), self.nt, x.shape[2], x.shape[3], device = x.get_device())
        p0 = torch.zeros((x.shape[0], len(self.trans_inds), x.shape[2], x.shape[3]), device = x.get_device())
        p1 = torch.zeros((x.shape[0], len(self.trans_inds), x.shape[2], x.shape[3]), device = x.get_device())
        eye = torch.eye(len(self.trans_inds), device = x.get_device())
        

        for j in range(self.nt):
            nab_p1_x = (-torch.roll(p1, 2, -2) + 16*torch.roll(p1, 1, -2)  - 30*p1 + 16*torch.roll(p1, -1, -2) - torch.roll(p1, -2, -2))/12
            nab_p1_y = (-torch.roll(p1, 2, -1) + 16*torch.roll(p1, 1, -1)  - 30*p1 + 16*torch.roll(p1, -1, -1) - torch.roll(p1, -2, -1))/12
            p = 2*p1  - p0 + (self.dt*x/self.dx)**2*(nab_p1_x + nab_p1_y)
            p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])
            p[:,:,self.trans_inds] = p[:,:,self.trans_inds] + (self.bg*self.dt/self.dx)**2*self.source_intentisty[j]*eye[None,:,:]
            p = p.reshape(p1.shape)
            out[:,:,j,:,:] = p
            p0 = p1
            p1 = p
        return out
    

class BornFoward(torch.nn.Module):
    def __init__(self, nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = 1.5, abc = 1.0, dev = None):
        super().__init__()
        times = torch.arange(nt)*dt
        self.source_intentisty = torch.sin(2*np.pi*fc*times)*torch.exp( -(times-t0)**2/(2*sgma**2))
        thetas = 2*np.pi*np.arange(nm)/nm
        meas_inds_x =(NX/2 + sR*np.cos(thetas)).astype(int)
        meas_inds_y =(NX/2 + sR*np.sin(thetas)).astype(int)
        self.meas_inds = NX*meas_inds_x + meas_inds_y

        self.nr = len(self.meas_inds//sr)

        self.dx = dx
        self.nx = nx
        self.NX = NX
        
        self.nt = nt
        self.dt = dt
        self.bg = bg
        
        self.abc = abc
        self.abc_pd = NX//2 - sR - 1
        self.abc_pad = (self.abc_pd, self.abc_pd, self.abc_pd, self.abc_pd)
        self.pd =(self.NX- self.nx)//2 - self.abc_pd
        self.pad = (self.pd, self.pd, self.pd, self.pd)

        helper = HelperFoward(nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = bg, abc = abc)

        with torch.no_grad():
            self.P0 = helper(bg*torch.ones((1,1, nx,nx)).to(dev))
            self.BG = self.bg*torch.ones((1, 1, nx, nx), device = dev)
            self.BG = torch.nn.functional.pad(self.BG, self.pad,    mode='constant', value=self.bg)
            self.BG = torch.nn.functional.pad(self.BG, self.abc_pad,    mode='constant', value=1e-6)
            



    def forward(self, x):

        x = self.bg/x
        x = torch.nn.functional.pad(x, self.pad,    mode='constant', value=1)
        x = torch.nn.functional.pad(x, self.abc_pad,    mode='constant', value=1)
        

        out = []
        p0 = torch.zeros((x.shape[0], self.P0.shape[1], x.shape[2], x.shape[3]), device = x.get_device())
        p1 = torch.zeros((x.shape[0], self.P0.shape[1], x.shape[2], x.shape[3]), device = x.get_device())
        

        for j in range(self.nt):
            nab_p1_x = (-torch.roll(p1, 2, -2) + 16*torch.roll(p1, 1, -2)  - 30*p1 + 16*torch.roll(p1, -1, -2) - torch.roll(p1, -2, -2))/12
            nab_p1_y = (-torch.roll(p1, 2, -1) + 16*torch.roll(p1, 1, -1)  - 30*p1 + 16*torch.roll(p1, -1, -1) - torch.roll(p1, -2, -1))/12
            p = 2*p1  - p0 + (self.dt*self.BG/self.dx)**2*(nab_p1_x + nab_p1_y)
            if j > 1:
                p = p + (self.BG/self.bg)**2*(1- x**2)*(self.P0[:,:,j,:,:] - 2*self.P0[:,:,j-1,:,:] + self.P0[:,:,j-2,:,:])
            p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])
            out.append( p[:,:, self.meas_inds])
            p = p.reshape(p1.shape)

            p0 = p1
            p1 = p
        return torch.stack(out, dim = 3)
    
class SEBornFoward(torch.nn.Module):
    def __init__(self, nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = 1.5, abc = 1.0, dev = None):
        super().__init__()
        times = torch.arange(nt)*dt
        self.source_intentisty = torch.sin(2*np.pi*fc*times)*torch.exp( -(times-t0)**2/(2*sgma**2))
        thetas = 2*np.pi*np.arange(nm)/nm
        meas_inds_x =(NX/2 + sR*np.cos(thetas)).astype(int)
        meas_inds_y =(NX/2 + sR*np.sin(thetas)).astype(int)
        self.meas_inds = NX*meas_inds_x + meas_inds_y

        self.nr = len(self.meas_inds//sr)

        self.dx = dx
        self.nx = nx
        self.NX = NX
        
        self.nt = nt
        self.dt = dt
        self.bg = bg
        
        self.abc = abc
        self.abc_pd = NX//2 - sR - 1
        self.abc_pad = (self.abc_pd, self.abc_pd, self.abc_pd, self.abc_pd)
        self.pd =(self.NX- self.nx)//2 - self.abc_pd
        self.pad = (self.pd, self.pd, self.pd, self.pd)

        helper = HelperFoward(nx, NX, dx, nt, dt, nm, sr, sR, fc, sgma, t0, bg = bg, abc = abc)

        with torch.no_grad():
            self.P0 = helper(bg*torch.ones((1,1, nx,nx)).to(dev))
            self.BG = self.bg*torch.ones((1, 1, nx, nx), device = dev)
            self.BG = torch.nn.functional.pad(self.BG, self.pad,    mode='constant', value=self.bg)
            self.BG = torch.nn.functional.pad(self.BG, self.abc_pad,    mode='constant', value=1e-6)
            



    def forward(self, x, se):
        


        x = self.bg/x
        x = torch.nn.functional.pad(x, self.pad,    mode='constant', value=1)
        x = torch.nn.functional.pad(x, self.abc_pad,    mode='constant', value=1)

        P0se = torch.unsqueeze(torch.einsum('ijkmn, j->ikmn', self.P0, se), 1)
        

        out = []
        p0 = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device = x.get_device())
        p1 = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device = x.get_device())
        

        for j in range(self.nt):
            nab_p1_x = (-torch.roll(p1, 2, -2) + 16*torch.roll(p1, 1, -2)  - 30*p1 + 16*torch.roll(p1, -1, -2) - torch.roll(p1, -2, -2))/12
            nab_p1_y = (-torch.roll(p1, 2, -1) + 16*torch.roll(p1, 1, -1)  - 30*p1 + 16*torch.roll(p1, -1, -1) - torch.roll(p1, -2, -1))/12
            p = 2*p1  - p0 + (self.dt*self.BG/self.dx)**2*(nab_p1_x + nab_p1_y)
            if j > 1:
                p = p + (self.BG/self.bg)**2*(1- x**2)*(P0se[:,:,j,:,:] - 2*P0se[:,:,j-1,:,:] + P0se[:,:,j-2,:,:])
            p = p.reshape(p.shape[0], p.shape[1], p.shape[2]*p.shape[3])
            out.append( p[:,:, self.meas_inds])
            p = p.reshape(p1.shape)

            p0 = p1
            p1 = p
        return torch.stack(out, dim = 3)
 
    
            




        



        