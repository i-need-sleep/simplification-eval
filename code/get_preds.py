import argparse

import utils.pred_simplification as pred_simplification
import utils.pretrain_stage_1_data_utils as pt_stage1_data_utils
import utils.globals as uglobals

def main(args):
    print(args)

    if args.method == 'save_txt':
        dataset = pt_stage1_data_utils.OpenWebTextDataset('save_txt', n_volumns=50, debug=False)
    elif args.method == 'gpt_turbo':
        dataset = pt_stage1_data_utils.OpenWebTextDataset('prompt', n_volumns=50, debug=False, txt_path=args.txt_path)
        pred_simplification.gptturbo_inference(dataset, out_dir=args.out_dir)
    elif args.method == 'gpt_curie':
        dataset = pt_stage1_data_utils.OpenWebTextDataset('prompt', n_volumns=50, debug=False, txt_path=args.txt_path)
        pred_simplification.gpt3_inference('curie', dataset, out_dir=args.out_dir)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='save_txt', type=str)  
    # parser.add_argument('--txt_path', default=f'{uglobals.STAGE2_RAW}/stage2_raw.en', type=str)  
    parser.add_argument('--txt_path', default=f'{uglobals.PROCESSED_DIR}/openwebtext/stage1_raw.en', type=str)  
    parser.add_argument('--out_dir', default=f'{uglobals.PROCESSED_DIR}/openwebtext', type=str)  

    args = parser.parse_args()

    main(args)