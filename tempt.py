            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
"DWRF_single_frame.py" 438L, 16338C                                                                                                                                         347,31        82%
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
                                                                                                                                                                            347,31        82%
            x, posterior_variance = self.q_posterior(x_0, x, u, u_next)
            noise = torch.randn_like(x)
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

gradient_accumulation_steps = 1
                                                                                                                                                                                                    347,31        83%
            x, posterior_variance = self.q_posterior(x_0, x, u, u_next)
            noise = torch.randn_like(x)
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

gradient_accumulation_steps = 1
                                                                                                                                                                                                    347,31        83%
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            # if i < steps*0.1:
            #      x_0 = x_0.clamp(-15, 15)
            x, posterior_variance = self.q_posterior(x_0, x, u, u_next)
            noise = torch.randn_like(x)
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

gradient_accumulation_steps = 1
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
device = accelerator.device

lrate = 10**-5
batch_size = 1
num_worker = 10
model_path="/home/user/Documents/Huaneng/V_2024_03_02/upper_15min_trained.pt"
                                                                                                                                                                                                                                   347,31        84%
        exp = torch.exp((self.u2lambda(u)-self.u2lambda(u_next))/2.)
        posterior_mean_coef1 = (1-exp)*self.u2alpha(u_next)
        posterior_mean_coef2 = exp*self.u2alpha(u_next)/self.u2alpha(u)
        posterior_mean = (posterior_mean_coef1.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_0 + (posterior_mean_coef2.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_t
        return posterior_mean

    @torch.inference_mode()
    def sample_ddpm(self, model, n, steps = 1000):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            # if i < steps*0.1:
            #      x_0 = x_0.clamp(-15, 15)
            x, posterior_variance = self.q_posterior(x_0, x, u, u_next)
            noise = torch.randn_like(x)
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

gradient_accumulation_steps = 1
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
device = accelerator.device

lrate = 10**-5
batch_size = 1
num_worker = 10
model_path="/home/user/Documents/Huaneng/V_2024_03_02/upper_15min_trained.pt"
dataset = PriorDataset(files)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

channel = 7
dim = 128
model = UNet3D(dim = dim, channels= channel, dim_mults=(1, 2, 4, 8))
size = [7, 48, 96, 96]


optimizer = optim.Adam(model.parameters(), lr=lrate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.01*lrate)
mse = nn.MSELoss()
diffusion = Diffusion(size=size , lambda_range = 20,  steps = 1000, device=device)
                                                                                                                                                                                                                                                                                                                                     347,31        87%

    def noise_images(self, x, u):
        eps = torch.randn_like(x).to(self.device)
        x_t = (self.u2alpha(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*x + (self.u2sigma(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*eps
        v = (self.u2alpha(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*eps - (self.u2sigma(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*x
        return x_t, eps, v

    def sample_timesteps(self, n):
        return torch.rand(n).to(self.device)


    def predict_start_from_v(self, x_t, u, v):
        sigma = (self.u2sigma(u).reshape([len(u)]+[1 for xx in x_t.shape[1:]]))
        alpha = (self.u2alpha(u).reshape([len(u)]+[1 for xx in x_t.shape[1:]]))
        return alpha*x_t-sigma*v

    def q_posterior(self, x_0, x_t, u, u_next):
        #Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        exp = torch.exp(self.u2lambda(u)-self.u2lambda(u_next))
        posterior_mean_coef1 = (1-exp)*self.u2alpha(u_next)
        posterior_mean_coef2 = exp*self.u2alpha(u_next)/self.u2alpha(u)
        posterior_mean = (posterior_mean_coef1.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_0 + (posterior_mean_coef2.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_t
        posterior_variance = (1-exp)*(self.u2sigma(u_next))*(self.u2sigma(u))
        return posterior_mean, posterior_variance

    def i_posterior(self, x_0, x_t, u, u_next):
        #Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        exp = torch.exp((self.u2lambda(u)-self.u2lambda(u_next))/2.)
        posterior_mean_coef1 = (1-exp)*self.u2alpha(u_next)
        posterior_mean_coef2 = exp*self.u2alpha(u_next)/self.u2alpha(u)
        posterior_mean = (posterior_mean_coef1.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_0 + (posterior_mean_coef2.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_t
        return posterior_mean

    @torch.inference_mode()
    def sample_ddpm(self, model, n, steps = 1000):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            # if i < steps*0.1:
            #      x_0 = x_0.clamp(-15, 15)
            x, posterior_variance = self.q_posterior(x_0, x, u, u_next)
            noise = torch.randn_like(x)
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

gradient_accumulation_steps = 1
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
device = accelerator.device

lrate = 10**-5
batch_size = 1
num_worker = 10
model_path="/home/user/Documents/Huaneng/V_2024_03_02/upper_15min_trained.pt"
dataset = PriorDataset(files)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

channel = 7
dim = 128
model = UNet3D(dim = dim, channels= channel, dim_mults=(1, 2, 4, 8))
size = [7, 48, 96, 96]


optimizer = optim.Adam(model.parameters(), lr=lrate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.01*lrate)
mse = nn.MSELoss()
diffusion = Diffusion(size=size , lambda_range = 20,  steps = 1000, device=device)

model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)

if os.path.isfile(model_path):
    print("loading")
    data = torch.load(model_path, map_location=device)
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    scheduler.load_state_dict(data['scheduler'])

global_step = 0
for epoch in range(1000):
    loss_running = 0.0
    pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
    for i, data in enumerate(pbar):
        data = data[:,:,random.randint(0,3),0:48,6:-6,4:-5]
        # print(data.shape)
        data = data.to(device=device, dtype=torch.float)
        # data = data[:,:,:,:-1]
        # print(data.shape)
        t = diffusion.sample_timesteps(data.shape[0])
        x_t, noise, v = diffusion.noise_images(data, t)
        predicted_v= model(x_t, t)
        loss = mse(predicted_v, v)
        loss_running = (loss_running * i + loss.item()) / (i + 1)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          347,31        95%
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv3d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv3d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time):
        # [tbatch,tchannel,thour,theight,tlength,twidth]= x.shape
        # x = x.reshape((tbatch,-1,theight,tlength,twidth))
        x = self.init_conv(x)
        r = x.clone()
        if time[0]<1:
            time = time *1000

        t = self.time_mlp(time)
        h = []
        #for block1, block2, attn, downsample in self.downs:
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        #for block1, block2, attn, upsample in self.ups:
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        # x = x.reshape((tbatch,tchannel,thour,theight,tlength,twidth))
        return x

class Diffusion:
    def __init__(self, size, lambda_range = 20, steps = 1000, device='cuda'):
        self.size = size
        self.device = device
        self.lambda_range = lambda_range
        self.steps = steps
        self.b = math.atan(math.exp(-lambda_range/2.))
        self.a = math.atan(math.exp(lambda_range/2))-math.atan(math.exp(-lambda_range/2.))
        self.bound = 0.9999

    def u2lambda(self,u):
        return -2*torch.log(torch.tan(self.a*u+self.b)).to(self.device)

    def u2alpha(self,u):
        return torch.sqrt(1./(1+torch.exp(-self.u2lambda(u)))).to(self.device)

    def u2sigma(self,u):
        return torch.sqrt(1.-1./(1+torch.exp(-self.u2lambda(u)))).to(self.device)

    def noise_images(self, x, u):
        eps = torch.randn_like(x).to(self.device)
        x_t = (self.u2alpha(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*x + (self.u2sigma(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*eps
        v = (self.u2alpha(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*eps - (self.u2sigma(u).reshape([len(u)]+[1 for xx in x.shape[1:]]))*x
        return x_t, eps, v

    def sample_timesteps(self, n):
        return torch.rand(n).to(self.device)


    def predict_start_from_v(self, x_t, u, v):
        sigma = (self.u2sigma(u).reshape([len(u)]+[1 for xx in x_t.shape[1:]]))
        alpha = (self.u2alpha(u).reshape([len(u)]+[1 for xx in x_t.shape[1:]]))
        return alpha*x_t-sigma*v

    def q_posterior(self, x_0, x_t, u, u_next):
        #Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        exp = torch.exp(self.u2lambda(u)-self.u2lambda(u_next))
        posterior_mean_coef1 = (1-exp)*self.u2alpha(u_next)
        posterior_mean_coef2 = exp*self.u2alpha(u_next)/self.u2alpha(u)
        posterior_mean = (posterior_mean_coef1.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_0 + (posterior_mean_coef2.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_t
        posterior_variance = (1-exp)*(self.u2sigma(u_next))*(self.u2sigma(u))
        return posterior_mean, posterior_variance

    def i_posterior(self, x_0, x_t, u, u_next):
        #Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        exp = torch.exp((self.u2lambda(u)-self.u2lambda(u_next))/2.)
        posterior_mean_coef1 = (1-exp)*self.u2alpha(u_next)
        posterior_mean_coef2 = exp*self.u2alpha(u_next)/self.u2alpha(u)
        posterior_mean = (posterior_mean_coef1.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_0 + (posterior_mean_coef2.reshape([len(u)]+[1 for xx in x_t.shape[1:]]))*x_t
        return posterior_mean

    @torch.inference_mode()
    def sample_ddpm(self, model, n, steps = 1000):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            # if i < steps*0.1:
            #      x_0 = x_0.clamp(-15, 15)
            x, posterior_variance = self.q_posterior(x_0, x, u, u_next)
            noise = torch.randn_like(x)
            if len(self.size)==3:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None]*noise
            elif len(self.size)==4:
                x = x + torch.sqrt(posterior_variance)[:,None,None,None,None]*noise
            if i % 100 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

    @torch.inference_mode()
    def sample_ddim(self, model, n, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq,torch.tensor([0]).to(self.device)))
        x = torch.randn([n]+self.size).to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(n).to(self.device) * u_seq[i]
            u_next = torch.ones(n).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            #if is_cond:
            #    #cond = cond.to(self.device)
            #    predicted_noise_cond = model_cond(x, t, cond)
            #    predicted_noise = torch.lerp(predicted_noise, predicted_noise_cond, cfg_scale)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            x= self.i_posterior(x_0, x, u, u_next)
            # if i % 10 ==0:
            #     print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x


    @torch.inference_mode()
    def sample_ddim_linear_cond(self, model, cond, vcond=0.1, steps = 100):
        u_seq = torch.arange(self.bound, 0, -1./steps).to(self.device)
        u_seq = torch.cat((u_seq, torch.tensor([0]).to(self.device)))
        x = torch.randn([len(cond)]+self.size).to(self.device)
        vcond=(torch.ones(1)*vcond).to(self.device)
        cond = cond.to(self.device)
        for i in tqdm(range(0, len(u_seq)-1), total=len(u_seq)):
            u = torch.ones(len(cond)).to(self.device) * u_seq[i]
            u_next = torch.ones(len(cond)).to(self.device) * u_seq[i+1]
            predicted_noise = model(x, u)
            x_0 = self.predict_start_from_v(x, u, predicted_noise)
            if i < steps*0.5:
                 var1 = (self.u2sigma(u))**2
                 var2 = vcond
                 ratio1=var2/(var1+var2)
                 ratio2=var1/(var1+var2)
                 x_0 = ratio1[:,None,None,None,None]*x_0+ratio2[:,None,None,None,None]*cond
                 x_0 = x_0.clamp(-15, 15)
            x= self.i_posterior(x_0, x, u, u_next)
            if i % 10 ==0:
                print([x.min(),x.max(),x_0.min(),x_0.max()])
        return x

gradient_accumulation_steps = 1
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
device = accelerator.device

lrate = 10**-5
batch_size = 1
num_worker = 10
model_path="/home/user/Documents/Huaneng/V_2024_03_02/upper_15min_trained.pt"
dataset = PriorDataset(files)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

channel = 7
dim = 128
model = UNet3D(dim = dim, channels= channel, dim_mults=(1, 2, 4, 8))
size = [7, 48, 96, 96]


optimizer = optim.Adam(model.parameters(), lr=lrate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.01*lrate)
mse = nn.MSELoss()
diffusion = Diffusion(size=size , lambda_range = 20,  steps = 1000, device=device)

model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)

if os.path.isfile(model_path):
    print("loading")
    data = torch.load(model_path, map_location=device)
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    scheduler.load_state_dict(data['scheduler'])

global_step = 0
for epoch in range(1000):
    loss_running = 0.0
    pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
    for i, data in enumerate(pbar):
        data = data[:,:,random.randint(0,3),0:48,6:-6,4:-5]
        # print(data.shape)
        data = data.to(device=device, dtype=torch.float)
        # data = data[:,:,:,:-1]
        # print(data.shape)
        t = diffusion.sample_timesteps(data.shape[0])
        x_t, noise, v = diffusion.noise_images(data, t)
        predicted_v= model(x_t, t)
        loss = mse(predicted_v, v)
        loss_running = (loss_running * i + loss.item()) / (i + 1)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        accelerator.wait_for_everyone()
        global_step += 1
        pbar.set_postfix(Epoch=epoch, MSE=loss.item(), MSE_running=loss_running, step=global_step)
        if global_step % 5000 == 0:
            data = {
                    'model': accelerator.get_state_dict(model),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                   }
            torch.save(data, model_path)
        accelerator.wait_for_everyone()
    scheduler.step()



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  217,31        Bot
