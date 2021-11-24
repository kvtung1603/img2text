from Vocab import Vocab
from dataset.aug import ImgAugTransform
from train import Train


if __name__ == "__main__":
    chars = '0Í+ẳ*kWUY&sOMỎ?ÔờỆồỌ°ỰV2ứcớẪh–iúẠ58ắ:öẵ)>ễự(Óữ<ỶP!;Ụ%DíŨầteấgõỂố}ẹỘ^âậỗỖŌỜKlỄ−\nQZpbảÝ_ỔặÁùEó,ừỸơ=zẨ4ẶịNợwỪ’ụỮằỵỏ7Ố6Ằ/ƯÌửàăạò-ĩỞỬỷÙẼTỨýể{ẫ.Ậ`AỴọXỳÕệ[$ãẢvI~èjLĂđ]dGẺỊ"3@ìRôẾnẩẦỹyJÈẸ#Éẻẽáưỡế ÃÚẴÒ\'ĨBCủ9SmêfỲẮréuũỈÀỒẤƠỉxộoẲỀÖÊỚởqFÂōỦ\\ềỠỢ1|ĐaổH'
    vocab = Vocab(chars)
    lmdb_path = "myDataset"
    root_dir = "../input/ocr-data/ocr_data"
    train_file = "train_file.txt"
    test_file = "test_file.txt"
    data_aug = ImgAugTransform()
    train = Train(chars, lmdb_path, root_dir, train_file, test_file, data_aug)
    train.train()
