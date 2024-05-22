import torch

# checkpoint_path = './checkpoints/sam_vit_b_01ec64.pth'
# checkpoint = torch.load(checkpoint_path)
# new_state_dict = {}
# for k, v in checkpoint.items():
#     # 修改键名，去掉 'image_encoder.' 前缀
#     if k.startswith('image_encoder.'):
#         new_key = k.replace('image_encoder.', '')
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
#
# torch.save(new_state_dict, checkpoint_path)
#
#
# modified_checkpoint = torch.load(checkpoint_path)
# print("Modified checkpoint keys after saving:", modified_checkpoint.keys())



checkpoint_path = './checkpoints/mask2former_r50_8xb2-lsj-50e_coco.pth'
# checkpoint = torch.load(checkpoint_path)
#
# new_state_dict = {}

# for k, v in checkpoint.items():
# # 修改键名，去掉 'image_encoder.' 前缀
#     if k.startswith('panoptic_head.'):
#         new_key = k.replace('panoptic_head.', '')
#     else:
#         new_key = k
#
#     new_state_dict[new_key] = v
#
# torch.save(new_state_dict, checkpoint_path)


modified_checkpoint = torch.load(checkpoint_path)
print("Modified checkpoint keys after saving:", modified_checkpoint.keys())


# with open('params1.txt','w') as f:
#     for k, v in checkpoint.items():
#         if k.startswith('backbone.'):
#             continue
#         else:
#             f.write(k)
#             f.write('\n')