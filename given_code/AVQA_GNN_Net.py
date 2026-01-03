class AVQA_GNN(nn.Module):
    def __init__(self, args, num_node_features=512, num_edge_features=512):
        super(AVQA_GNN, self).__init__()
        # omit some pieces of codes
        # ...
        self.wv = Linear(out_channels, out_channels)
        self.wq = Parameter(torch.zeros(1, 10))
        self.joint_linear = Bilinear(out_channels, out_channels, out_channels)
        self.lin_a = Sequential(
                Linear(128, 512),
                ReLU(),
                Linear(512, 512)
            )
        # omit some pieces of codes
        # ...
    
    def forward(self, audio_feat, visual_feat, question_feat, sg_data, qg_data):
        # omit some pieces of codes
        # ...
        sim = torch.bmm(video, query.permute(0, 2, 1)) # [B, n:10, q]
        temporature = 1.0
        v_joint = torch.bmm((sim/temporature).permute(0, 2, 1), video) # [B, q, 512]
        v_joint = self.wv(v_joint).mean(dim=1) # [B, 512]
        q_joint = torch.bmm((sim/temporature), query) # [B, 10, 512]
        q_joint = (self.wq.unsqueeze(-1) * q_joint).sum(dim=1) # [B, 512]
        vq_joint = self.joint_linear(v_joint, q_joint) # [B, 512]
        # omit some pieces of codes
        # ...

class MgA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MgA, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 3)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 5)

    def forward(self, input):
        conv_1 = self.conv1(input).permute(0, 2, 1) # [B, 10, 512]
        conv_2 = self.conv2(input).permute(0, 2, 1) # [B, 8, 512]
        conv_3 = self.conv3(input).permute(0, 2, 1) # [B, 6, 512]

        return conv_1, conv_2, conv_3

class CrossAttention(nn.Module):
    def __init__(self, dim=512):
        super(CrossAttention, self).__init__()
        self.w = Linear(dim, dim)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, audio_conv_list, video_conv_list):
        (B, T, C) = audio_conv_list[0].shape
        f_v = [[], [], []]
        f_a = [[], [], []]
        for i, video_conv in enumerate(video_conv_list):
            for j, audio_conv in enumerate(audio_conv_list):
                a_ij = F.softmax(torch.bmm(self.w(video_conv), audio_conv.permute(0, 2, 1)) / torch.sqrt(torch.tensor(video_conv.shape[-1]))) # [4, 10, 10]
                f_v[i].append(torch.bmm(a_ij, audio_conv))
                f_a[j].append(torch.bmm(a_ij.permute(0, 2, 1), video_conv))

        return f_v, f_a

# The code below gives you a sense of the multi-grained alignment part in the pipeline.
class AVQA_GNN(nn.Module):
    def __init__(self, args, num_node_features=512, num_edge_features=512):
        super(AVQA_GNN, self).__init__()
        # omit some pieces of codes
        # ...
        self.mga_v = MgA(out_channels, out_channels)
        self.mga_a = MgA(out_channels, out_channels)
        self.cross_attn = CrossAttention(out_channels)
        # omit some pieces of codes
        # ...
    
    def forward(self, audio_feat, visual_feat, question_feat, sg_data, qg_data):
        # omit some pieces of codes
        # ...
        audio_conv_1, audio_conv_2, audio_conv_3 = self.mga_a(audio_feat.permute(0, 2, 1)) # [B, l_a, 512]
        video_conv_1, video_conv_2, video_conv_3 = self.mga_v(visual_feat.permute(0, 2, 1)) # [B, l_v, 512]

        audio_conv_list = [audio_conv_1, audio_conv_2, audio_conv_3]
        video_conv_list = [video_conv_1, video_conv_2, video_conv_3]

        f_v, f_a = self.cross_attn(audio_conv_list, video_conv_list) # [3, 3, B, 10, 512]
        # omit some pieces of codes
        # ...

# Firstly match and combine the multi-scale visual-audio representations within the same kernel size.
class HierarchicalMatch(nn.Module):
    def __init__(self, N=3, dim=512):
        super(HierarchicalMatch, self).__init__()
        self.N = N
        self.dim = dim

    def forward(self, joint, f, B):
        N = self.N
        dim = self.dim
        b = torch.zeros(N, N, B).to('cuda')
        for i, f_i in enumerate(f):
            for j, f_ij in enumerate(f_i):
                f_ij_tmp = torch.mean(f_ij, dim=1) # [B, 512]
                b[i][j] = torch.bmm(joint.unsqueeze(1), f_ij_tmp.unsqueeze(-1)).squeeze() / torch.stack([torch.bmm(joint.unsqueeze(1), f[i][r].mean(dim=1).unsqueeze(-1)).squeeze() for r in range(N)]).sum(dim=0) # [B]

        f_ii = []
        for i, f_i in enumerate(f):
            f_ii.append(torch.stack([b[i][j][:, None, None] * f_ij for j, f_ij in enumerate(f_i)]).sum(dim=0))

        lambda_i = torch.zeros(N, B).to('cuda')
        for i, f_i in enumerate(f_ii):
            f_i_tmp = torch.mean(f_i, dim=1) # [B, 512]
            lambda_i[i] = torch.bmm(joint.unsqueeze(1), f_i_tmp.unsqueeze(-1)).squeeze() / torch.stack([torch.bmm(joint.unsqueeze(1), f_ii[r].mean(dim=1).unsqueeze(-1)).squeeze() for r in range(N)]).sum(dim=0)

        return torch.stack([lambda_i[i][:, None] * f_i.mean(dim=1) for i, f_i in enumerate(f_ii)]).sum(dim=0) # [B, 512]

# Then combine the representations for different kernel sizes into one multi-modal representation $F_M$.
class AVQA_GNN(nn.Module):
    def __init__(self, args, num_node_features=512, num_edge_features=512):
        super(AVQA_GNN, self).__init__()
        # omit some pieces of codes
        # ...
        self.match_v = HierarchicalMatch(N=3, dim=out_channels)
        self.match_a = HierarchicalMatch(N=3, dim=out_channels)
        self.tanh_avq = nn.Tanh()
        # omit some pieces of codes
        # ...
    
    def forward(self, audio_feat, visual_feat, question_feat, sg_data, qg_data):
        # omit some pieces of codes
        # ...
        f_v = self.match_v(vq_joint, f_v, B) # [B, 512]
        f_a = self.match_a(vq_joint, f_a, B)

        z_v = F.sigmoid(vq_joint * f_v) # [B, 512]
        z_a = F.sigmoid(vq_joint * f_a)

        f_m = z_v * f_v + z_a * f_a # [B, 512]

        avq_feat = f_m * question_feat.squeeze(1) # [B, 512]
        avq_feat = self.tanh_avq(avq_feat)
        # omit some pieces of codes
        # ...