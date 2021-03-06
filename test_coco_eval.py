from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

def main(args):
	coco = COCO(args.annotations_path)
	coco_res = coco.loadRes(args.results_json_path)

	coco_eval = COCOEvalCap(coco, coco_res)

	coco_eval.params['image_id'] = coco_res.getImgIds()

	coco_eval.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_json_path', type=str, required=True)
    parser.add_argument('--annotations_path', type=str, default='annotations/captions_val2014.json')

    args = parser.parse_args()
    print(args)
    main(args)



