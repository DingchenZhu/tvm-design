import math
from torch import nn
import torch
from thop import profile
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=32, s=8, m=4):
        super(FSRCNN, self).__init__()

        self.quant_bit_offset = 6

        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=3, padding=3//2),
            nn.PReLU(d)
        )

        self.mid_part_1 = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m//2):
            self.mid_part_1.extend([DeformableConv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)])
          

        self.mid_part_2 = nn.ModuleList()
        for _ in range(m//2):
            self.mid_part_2.extend([DeformableConv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)])
        self.mid_part_2.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part_1 = nn.Sequential(*self.mid_part_1)
        self.mid_part_2 = nn.Sequential(*self.mid_part_2)
        self.last_part = UpsampleBlock(d, num_channels, scale_factor=scale_factor)

        self._initialize_weights()
        self.scale = scale_factor

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part_1:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part_2:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

    # def forward(self, x):
    #     x = self.first_part(x)
    #     x = self.mid_part_1(x)
    #     x = self.mid_part_2(x)
    #     x = self.last_part(x) 
    #     return x
    def forward(self, x):
        for layer in self.first_part:
            x = layer(x)

        for layer in self.mid_part_1:
            if isinstance(layer, DeformableConv2d):
                layer.quant_bit = self.quant_bit_offset
                x = layer(x)
            else:
                x = layer(x)
        
        for layer in self.mid_part_2:
            if isinstance(layer, DeformableConv2d):
                layer.quant_bit = self.quant_bit_offset
                x = layer(x)
            else:
                x = layer(x)

        x = self.last_part(x)


        return x


    
class UpsampleBlock(nn.Module):
    def __init__(self, d, num_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        upsample_channels = (scale_factor ** 2) * num_channels
        
        self.conv = nn.Conv2d(d, upsample_channels, kernel_size=3, 
                             padding=3//2, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x
    
class OffsetGenerator(nn.Module):
    def __init__(self, C, reuse=4):
        super(OffsetGenerator, self).__init__()
        self.reuse = reuse
        self.off2d_1 = nn.Sequential(
            nn.AvgPool2d(reuse),
            nn.Conv2d(C, 2 * 9, 3, padding=1, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        offset = self.off2d_1(x)
        # print(offset.shape)
        offset = torch.clamp(offset, -1, 1)
        offset = torch.repeat_interleave(offset, repeats=self.reuse, dim=2)
        offset = torch.repeat_interleave(offset, repeats=self.reuse, dim=3)

        # 统一尺寸处理（推荐用这个）
        H, W = x.shape[2], x.shape[3]
        offset = F.pad(offset, [0, W - offset.shape[3], 0, H - offset.shape[2]], mode='replicate')
        offset = offset[:, :, :H, :W]
        # print((offset.abs() > 1).float().mean().item())
        return offset
    
class DeformableConv2d(nn.Module):
    def __init__(self, s, out_channels, kernel_size=3, padding=1):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.quant_bit = 4
        
        # 偏移量生成器
        self.off2d = OffsetGenerator(s, 4)
        # self.off2d = nn.Conv2d(s, 2 * 9, 3, padding=1, bias=True)
        # 可变形卷积
        self.dconv = nn.Conv2d(s, out_channels, kernel_size, 
                              padding=padding)

    def deform_conv2d_custom(self, input, offset, weight, bias, stride=1, padding=1):
        N, C_in, H_in, W_in = input.shape
        C_out, _, Kh, Kw = weight.shape
        
        H_out = (H_in + 2*padding - Kh)//stride + 1
        W_out = (W_in + 2*padding - Kw)//stride + 1
        input_pad = F.pad(input, (padding, padding, padding, padding))
    
        output = torch.zeros((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)

        def get_value(yy, xx):
            if 0 <= yy < H_in + 2*padding and 0 <= xx < W_in + 2*padding:
                return input_pad[n, ic, yy, xx]
            else:
                return 0.0

        print(N, C_in, H_in, W_in, C_out, Kh, Kw)
        
        # for循环遍历 这部分运行速度似乎很慢
        for n in range(N):
            for oc in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        # print(n, oc, h, w)
                        val = 0.0
                        for ic in range(C_in):
                            for ky in range(Kh):
                                for kx in range(Kw):
                                    # 原始采样点，以左上角为基点
                                    y = h*stride + ky
                                    x = w*stride + kx
                                    
                                    # 对应的offset
                                    off_y = offset[n, 2*(ky*Kw + kx), h, w]
                                    off_x = offset[n, 2*(ky*Kw + kx)+1, h, w]
                                    
                                    y_off = y + off_y
                                    x_off = x + off_x
                                    
                                    # 双线性插值
                                    y0 = torch.floor(y_off).long()
                                    x0 = torch.floor(x_off).long()
                                    y1 = y0 + 1
                                    x1 = x0 + 1
                                    
                                    ly = y_off - y0.float()
                                    lx = x_off - x0.float()
                                    hy = 1 - ly
                                    hx = 1 - lx
                                
                                    v00 = get_value(y0, x0)
                                    v01 = get_value(y0, x1)
                                    v10 = get_value(y1, x0)
                                    v11 = get_value(y1, x1)
                                    
                                    val_sample = (hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11)
                                    val += val_sample * weight[oc, ic, ky, kx]
                        
                        if bias is not None:
                            val += bias[oc]
                        output[n, oc, h, w] = val
        return output

    def deform_conv2d_fast(self, input, offset, weight, bias, stride=1, padding=1):
        N, C_in, H_in, W_in = input.shape
        C_out, _, Kh, Kw = weight.shape
        
        H_out = (H_in + 2*padding - Kh)//stride + 1
        W_out = (W_in + 2*padding - Kw)//stride + 1
        input_pad = F.pad(input, (padding, padding, padding, padding))
    
        output = torch.zeros((N, C_out, H_out, W_out), device=input.device, dtype=input.dtype)

        def get_value(yy, xx):
            if 0 <= yy < H_in + 2*padding and 0 <= xx < W_in + 2*padding:
                return input_pad[n, ic, yy, xx]
            else:
                return 0.0

        print(N, C_in, H_in, W_in, C_out, Kh, Kw)
        
        for n in range(N):
            for oc in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        # print(n, oc, h, w)
                        val = 0.0
                        for ic in range(C_in):
                            for ky in range(Kh):
                                for kx in range(Kw):
    
                                    y = h*stride + ky
                                    x = w*stride + kx
                                    
                                    off_y = offset[n, 2*(ky*Kw + kx), h, w]
                                    off_x = offset[n, 2*(ky*Kw + kx)+1, h, w]
                                    
                                    y_off = y + off_y
                                    x_off = x + off_x

                                    y0 = torch.floor(y_off).long()
                                    x0 = torch.floor(x_off).long()
                                    y1 = y0 + 1
                                    x1 = x0 + 1
                                    
                                    ly = y_off - y0.float()
                                    lx = x_off - x0.float()
                                    hy = 1 - ly
                                    hx = 1 - lx
                                
                                    v00 = get_value(y0, x0)
                                    v01 = get_value(y0, x1)
                                    v10 = get_value(y1, x0)
                                    v11 = get_value(y1, x1)
                                    
                                    val_sample = (hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11)
                                    val += val_sample * weight[oc, ic, ky, kx]
                        
                        if bias is not None:
                            val += bias[oc]
                        output[n, oc, h, w] = val
        return output

    def forward(self, x):
        # 生成偏移量
        offset = self.off2d(x)
        # x = deform_conv2d(input=x, offset=offset, weight=self.dconv.weight, bias=self.dconv.bias, stride=(1,1),
        #                     padding=(1,1), dilation=(1,1), mask=None)
        # x = self.deform_conv2d_custom(input=x, offset=offset, weight=self.dconv.weight, bias=self.dconv.bias, stride=1,
        #                     padding=1)
        
        # offset移位裁剪
        # offset = torch.floor(offset * (2**self.quant_bit)) / (2**self.quant_bit)

        # 验证一致性
        y1 = deform_conv2d(input=x, offset=offset, weight=self.dconv.weight, bias=self.dconv.bias, stride=(1,1),padding=(1,1), dilation=(1,1), mask=None)
        # y2 = self.deform_conv2d_custom(input=x, offset=offset, weight=self.dconv.weight, bias=self.dconv.bias, stride=1,padding=1)
        # print(torch.allclose(y1, y2, atol=1e-4))

        return y1



if __name__ == '__main__':
    input = torch.randn(1,1,540,960)
    model = FSRCNN(2, num_channels=1, d=32, s=8, m=4)
    data=profile(model,inputs=(input,))
    x0= torch.randn(1,1,6,6)
    print(x0)
    x = nn.AvgPool2d(4)(x0)
    print(x0[:,:,0:4,0:4])
    x1 = nn.AvgPool2d(4)(x0[:,:,0:4,0:4])
    print(x1.shape)
    print(x,x1)
    print(data)
    print(model)