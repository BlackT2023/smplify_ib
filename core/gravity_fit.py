import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.vis.smpl_vis import batch_optimization_render

def fit(args, datasets, smplify, device):

    opt_loader = DataLoader(
        dataset=datasets,
        batch_size=args.batch_size,
        shuffle=False
    )

    image_path = datasets.get_curr_image_path()

    num_steps_total = len(datasets) // args.batch_size

    final_loss_list = []
    final_loss_dict = {}
    output_list = {}

    depth = []

    for step, batch in enumerate(tqdm(opt_loader, desc='Epoch ' + str(1),
                                      total=num_steps_total)):

        if step not in [8, 9, 10]:
            continue
        print(step)
        batch = {k: v.type(torch.float32).squeeze().detach().to(device).requires_grad_(False) for k, v in batch.items()}

        output, loss_dict, final_metrics = smplify(
            init_pose=batch['est_pose'],
            init_betas=batch['est_betas'],
            init_trans=batch['trans'],
            keypoints_2d=batch['keypoints_pix'],
            depth_bed=batch['depth_bed'],
            gt_height=batch['height'],
            pressure=batch['binary_pressure'],
            sensor_position=batch['sensor_position'],
            bed_corner_shift=batch['bed_corner_shift'],
            gt_depth=batch['depth_dis_array'],
        )
        # if datasets.stage == 2:
        #     batch_optimization_render(args, image_path, batch['curr_frame_idx'].type(torch.int32).cpu().detach().numpy(),
        #                               output['verts'], output['faces'])

        datasets.save_opt_results(batch['curr_frame_idx'].type(torch.int32).cpu().detach().numpy(),
                                  output['pose'],
                                  output['betas'],
                                  output['trans']
                                  )

        process(final_metrics, loss_dict, output, final_loss_list, final_loss_dict, output_list)

        depth.extend(np.max(output['verts'][:, :, 2], axis=1).tolist())

            # if step == 4:
            #     break
    # datasets.save_db()

    return output_list, final_loss_list, final_loss_dict

