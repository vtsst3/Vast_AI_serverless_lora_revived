import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

# 追加
import timm
from torchvision import transforms
from safetensors.torch import load_file as load_safetensors

import library.train_util as train_util
from library.utils import setup_logging, pil_resize

setup_logging()
import logging

logger = logging.getLogger(__name__)

# from wd14 tagger
IMAGE_SIZE = 448

# wd-v1-4-swinv2-tagger-v2 / wd-v1-4-vit-tagger / wd-v1-4-vit-tagger-v2/ wd-v1-4-convnext-tagger / wd-v1-4-convnext-tagger-v2
DEFAULT_WD14_TAGGER_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
# 新しいモデルの定義
EVA02_TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# 旧モデルのファイル定義
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
FILES_ONNX = ["model.onnx"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = "selected_tags.csv"

# 新モデルのファイル定義
SAFETENSORS_FILE = "model.safetensors"
ONNX_FILE = "model.onnx"


def preprocess_image_tf(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode="constant", constant_values=255)

    if size > IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_AREA)
    else:
        image = pil_resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype(np.float32)
    return image

def preprocess_image_pt(image):
    # timmモデル用の前処理
    # https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3#preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(image)


class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, is_pt_model):
        self.images = image_paths
        self.is_pt_model = is_pt_model

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            if self.is_pt_model:
                image = preprocess_image_pt(image)
            else:
                image = preprocess_image_tf(image)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (image, img_path)


def collate_fn_remove_corrupted(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def find_safetensors_file(model_dir):
    """指定されたディレクトリ内の.safetensorsファイルを検索する"""
    for file in os.listdir(model_dir):
        if file.endswith(".safetensors"):
            return file
    return None

def find_csv_file(model_dir):
    """指定されたディレクトリ内の.csvファイルを検索する"""
    for file in os.listdir(model_dir):
        if file.endswith(".csv"):
            return file
    return None

def find_onnx_file(model_dir):
    """指定されたディレクトリ内の.onnxファイルを検索する"""
    for file in os.listdir(model_dir):
        if file.endswith(".onnx"):
            return file
    return None

def main(args):
    # パス解決
    onnx_model_path = args.onnx_model_path
    csv_path = args.csv_path
    model_location = args.model_dir

    if not onnx_model_path and model_location:
        found_onnx = find_onnx_file(model_location)
        if found_onnx:
            onnx_model_path = os.path.join(model_location, found_onnx)

    if not csv_path and model_location:
        found_csv = find_csv_file(model_location)
        if found_csv:
            csv_path = os.path.join(model_location, found_csv)
    
    # is_pt_modelの判定を修正
    safetensors_file = None
    if model_location:
        found_sf = find_safetensors_file(model_location)
        if found_sf:
            safetensors_file = os.path.join(model_location, found_sf)
    
    is_pt_model = (safetensors_file is not None) and (not args.onnx)

    # モデルを読み込む
    if args.onnx or onnx_model_path:
        if not onnx_model_path:
            raise FileNotFoundError("ONNX model not found. Use --onnx_model_path or --model_dir.")
        logger.info("Loading ONNX model")
        logger.info(f"Creating ONNX InferenceSession from {onnx_model_path}")
        model = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    elif is_pt_model:
        logger.info(f"Loading PyTorch model (timm) from {safetensors_file}")
        model_path = safetensors_file
        # エラーメッセージに基づき、正しいクラス数に修正
        model = timm.create_model('eva02_large_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=False, num_classes=10861)
        state_dict = load_safetensors(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        if torch.cuda.is_available():
            model = model.to('cuda')
    else:
        from tensorflow.keras.models import load_model
        logger.info("Loading TensorFlow Keras model")
        model = load_model(f"{model_location}")

    if not csv_path:
        raise FileNotFoundError("CSV file not found. Use --csv_path or --model_dir.")
    logger.info(f"Loading tags from {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]
        rows = line[1:]
    
    # --- DEBUG ---
    logger.info(f"Loaded CSV file from: {csv_path}")
    logger.info(f"CSV header: {header}")
    logger.info(f"CSV first 5 rows: {rows[:5]}")
    # --- END DEBUG ---
    
    assert header[0] == "tag_id" and header[1] == "name", f"unexpected csv format: {header}"

    tag_names = [row[1] for row in rows] # 全タグを順序維持して取得

    # v3モデルはcategoryがないので調整
    if len(header) < 3:
        rating_tags = []  # v3にはレーティングタグなし
        # 後方互換性や他の部分での利用のためにリストを作成しておく
        general_tags = [name for name in tag_names if '_' not in name]
        character_tags = [name for name in tag_names if '_' in name]
    else:
        rating_tags = [row[1] for row in rows if row[2] == "9"]
        general_tags = [row[1] for row in rows if row[2] == "0"]
        character_tags = [row[1] for row in rows if row[2] == "4"]

    # (タグ前処理のコードは変更なし)
    # ... (省略) ...

    train_data_dir_path = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    logger.info(f"found {len(image_paths)} images.")

    tag_freq = {}
    caption_separator = args.caption_separator
    stripped_caption_separator = caption_separator.strip()
    undesired_tags = {tag.strip() for tag in args.undesired_tags.split(stripped_caption_separator) if tag.strip()}
    always_first_tags = [tag for tag in args.always_first_tags.split(stripped_caption_separator) if tag.strip()] if args.always_first_tags else None

    def run_batch(path_imgs):
        imgs_tensor = torch.stack([im for _, im in path_imgs]) if is_pt_model else np.array([im for _, im in path_imgs])

        if args.onnx:
            # ONNXはTFベースの前処理なので、numpy配列のまま
            imgs_np = np.array([im for _, im in path_imgs])
            input_name = model.get_inputs()[0].name
            label_name = model.get_outputs()[0].name
            
            probs = model.run([label_name], {input_name: imgs_np})[0]
        elif is_pt_model:
            with torch.no_grad():
                if torch.cuda.is_available():
                    imgs_tensor = imgs_tensor.to('cuda')
                probs = model(imgs_tensor)
                probs = torch.sigmoid(probs).cpu().numpy()
        else:
            probs = model(imgs_tensor, training=False)
            probs = probs.numpy()

        for (image_path, _), prob in zip(path_imgs, probs):
            # (タグ結合処理のコードは変更なし)
            # ... (省略) ...
            combined_tags = []
            
            # v3モデルはratingタグがないので、generalとcharacterのみ
            if is_pt_model:
                # is_pt_modelの場合、tag_namesリストはCSVの順序通りになっている
                for i, p in enumerate(prob):
                    tag_name = tag_names[i]
                    # タグ名にアンダースコアが含まれるかでキャラクタータグかどうかを判定
                    is_character = '_' in tag_name
                    
                    threshold = args.character_threshold if is_character else args.general_threshold
                    
                    if p >= threshold:
                        if tag_name not in undesired_tags:
                            combined_tags.append(tag_name)
            else: # 古いモデルの処理 (ONNXはこちらを通る)
                for i, p in enumerate(prob[4:]):
                    if i < len(general_tags) and p >= args.general_threshold:
                        tag_name = general_tags[i]
                        if tag_name not in undesired_tags:
                            combined_tags.append(tag_name)
                    elif i >= len(general_tags) and p >= args.character_threshold:
                        tag_name = character_tags[i - len(general_tags)]
                        if tag_name not in undesired_tags:
                            combined_tags.append(tag_name)
                if args.use_rating_tags or args.use_rating_tags_as_last_tag:
                    rating_index = prob[:4].argmax()
                    found_rating = rating_tags[rating_index]
                    if found_rating not in undesired_tags:
                        if args.use_rating_tags:
                            combined_tags.insert(0, found_rating)
                        else:
                            combined_tags.append(found_rating)

            if always_first_tags:
                for tag in always_first_tags:
                    if tag in combined_tags:
                        combined_tags.remove(tag)
                        combined_tags.insert(0, tag)

            tag_text = caption_separator.join(combined_tags)

            caption_file = os.path.splitext(image_path)[0] + args.caption_extension
            with open(caption_file, "wt", encoding="utf-8") as f:
                f.write(tag_text + "\n")

    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths, is_pt_model)
        data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data_item in data_entry:
            if data_item is None:
                continue
            image, image_path = data_item
            if image is None:
                try:
                    image = Image.open(image_path).convert("RGB")
                    if is_pt_model:
                        image = preprocess_image_pt(image)
                    else:
                        image = preprocess_image_tf(image)
                except Exception as e:
                    logger.error(f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))
            if len(b_imgs) >= args.batch_size:
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        run_batch(b_imgs)

    logger.info("done!")

# (setup_parserと__main__は変更なし)
# ... (省略) ...
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--model_dir", type=str, default=None, help="directory containing wd14 tagger model / wd14 taggerのモデルを格納したディレクトリ")
    parser.add_argument("--onnx_model_path", type=str, default=None, help="path to onnx model file / onnxモデルファイルへのパス")
    parser.add_argument("--csv_path", type=str, default=None, help="path to tags csv file / タグのcsvファイルへのパス")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=None, help="enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）")
    parser.add_argument("--caption_extention", type=str, default=None, help="extension of caption file (for backward compatibility) / 出力されるキャプションファイルの拡張子（スペルミスしていたのを残してあります）")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子")
    parser.add_argument("--thresh", type=float, default=0.35, help="threshold of confidence to add a tag / タグを追加するか判定する閾値")
    parser.add_argument("--general_threshold", type=float, default=None, help="threshold of confidence to add a tag for general category, same as --thresh if omitted / generalカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ")
    parser.add_argument("--character_threshold", type=float, default=None, help="threshold of confidence to add a tag for character category, same as --thres if omitted / characterカテゴリのタグを追加するための確信度の閾値、省略時は --thresh と同じ")
    parser.add_argument("--recursive", action="store_true", help="search for images in subfolders recursively / サブフォルダを再帰的に検索する")
    parser.add_argument("--remove_underscore", action="store_true", help="replace underscores with spaces in the output tags / 出力されるタグのアンダースコアをスペースに置き換える")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--undesired_tags", type=str, default="", help="comma-separated list of undesired tags to remove from the output / 出力から除外したいタグのカンマ区切りのリスト")
    parser.add_argument("--frequency_tags", action="store_true", help="Show frequency of tags for images / タグの出現頻度を表示する")
    parser.add_argument("--onnx", action="store_true", help="use onnx model for inference / onnxモデルを推論に使用する")
    parser.add_argument("--append_tags", action="store_true", help="Append captions instead of overwriting / 上書きではなくキャプションを追記する")
    parser.add_argument("--use_rating_tags", action="store_true", help="Adds rating tags as the first tag / レーティングタグを最初のタグとして追加する")
    parser.add_argument("--use_rating_tags_as_last_tag", action="store_true", help="Adds rating tags as the last tag / レーティングタグを最後のタグとして追加する")
    parser.add_argument("--character_tags_first", action="store_true", help="Always inserts character tags before the general tags / characterタグを常にgeneralタグの前に出力する")
    parser.add_argument("--always_first_tags", type=str, default=None, help="comma-separated list of tags to always put at the beginning, e.g. `1girl,1boy` / 必ず先頭に置くタグのカンマ区切りリスト、例 : `1girl,1boy`")
    parser.add_argument("--caption_separator", type=str, default=", ", help="Separator for captions, include space if needed / キャプションの区切り文字、必要ならスペースを含めてください")
    parser.add_argument("--tag_replacement", type=str, default=None, help="tag replacement in the format of `source1,target1;source2,target2; ...`. Escape `,` and `;` with `\`. e.g. `tag1,tag2;tag3,tag4` / タグの置換を `置換元1,置換先1;置換元2,置換先2; ...`で指定する。`\` で `,` と `;` をエスケープできる。例: `tag1,tag2;tag3,tag4`")
    parser.add_argument("--character_tag_expand", action="store_true", help="expand tag tail parenthesis to another tag for character tags. `chara_name_(series)` becomes `chara_name, series` / キャラクタタグの末尾の括弧を別のタグに展開する。`chara_name_(series)` は `chara_name, series` になる")
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh
    main(args)
