import torch
# takes in attention heads for queries and key and applies rope 
def build_rope_table(head_dim, theta, max_seq, dtype): 
    positions = torch.tensor([i for i in range(max_seq)], dtype = torch.float32).unsqueeze(0) # [1, max_seq]
    #inv_freq = torch.tensor([pow(theta,((-2*i)/head_dim)) for i in range(head_dim//2)], dtype=torch.float32).unsqueeze(0) #[1, 64]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)).unsqueeze(0) # [1, head_dim//2]
    freq = positions.T @ inv_freq
    cos_table = torch.concat((torch.cos(freq), torch.cos(freq)), dim=1) # [10, 128]
    sin_table = torch.concat((torch.sin(freq), torch.sin(freq)), dim=1)
    return cos_table.to(dtype), sin_table.to(dtype)


def apply_rope(x, cos, sin, start): 
    # x represents query or attention heads we apply RoPE to is of size [seq, num_heads, head_dim]
    max_seq = cos.shape[0]
    if start + x.shape[0] > max_seq:
        raise ValueError("Cannot exceed maximum context length")

    head_dim = x.shape[-1]
    seq = x.shape[0]
    p1 = x[:,:,0:head_dim//2]
    p2 = x[:,:,head_dim//2:]

    rot_head = torch.concat((-p2, p1), dim=-1)
    head_cos = x * cos[start:start+seq, None,:]
    head_sin = rot_head * sin[start:start+seq, None,:]
    res = head_cos + head_sin
    #print(res.shape)
    return res




# tensor = torch.ones((10, 128))
# print(tensor.shape)
# print(tensor)
# apply_rope(tensor, 100000)
