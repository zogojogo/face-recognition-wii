from src.feature_extract import extract_feature, NED
from models.inception_resnet import InceptionResnetV1
import time

def verification(input_image, enroll_image, model, mode):
  embed1 = extract_feature(input_image, model, mode)
  embed2 = extract_feature(enroll_image, model, mode)
  return 1-NED(embed1, embed2)

def predict_verif(input_image, enroll_image, model, mode):
    conf = verification(input_image, enroll_image, model, mode)
    output = 1 if conf > 0.75 else 0
    return conf, output

if __name__ == "__main__":
    model = InceptionResnetV1(pretrained='vggface2')
    start_time = time.time()
    print(verification("./uploads/1455022848752_0.jpg", "./new_data/25/25_1.jpg", model, 'x'))
    print(time.time() - start_time)