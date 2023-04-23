import argparse

import utils.pred_simplification as pred_simplification
import utils.pretrain_stage_1_data_utils as pt_stage1_data_utils

def main(args):
    print(args)

    if args.method == 'gpt_turbo':
        dataset = pt_stage1_data_utils.OpenWebTextDataset('prompt', 6, debug=False)
        pred_simplification.gptturbo_inference(dataset)
    elif args.method == 'gpt_curie':
        dataset = pt_stage1_data_utils.OpenWebTextDataset('prompt', 6, debug=False)
        pred_simplification.gpt3_inference('curie', dataset)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', default='gpt_turbo', type=str)  

    args = parser.parse_args()

    main(args)
