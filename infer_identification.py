from src.feature_extract import embed2dict, extract_feature, NED
from models.inception_resnet import InceptionResnetV1
import time

def identification(img_input, model, embeddings, mode):
    result = {}
    input_embedding = extract_feature(img_input, model, mode)
    for img_path, img_embedding in embeddings.items():
        result[img_path] = 1-NED(input_embedding, img_embedding)
    return result

def get_name_id(img_path, model, embeddings, mode):
  result = identification(img_path, model, embeddings, mode)
  result_sorted = sorted(result.items(), key=lambda x: x[1], reverse=True)
  conf = result_sorted[0][1]
  return conf, result_sorted[0][0].split("/")[-2]

if __name__ == "__main__":
    model = InceptionResnetV1(pretrained='vggface2')
    embeddings = embed2dict("./new_data/", model)
    start_time = time.time()

    
    conf, prediction = get_name_id("./uploads/1455022848752_0.jpg", model, embeddings, '3')


    print(time.time() - start_time)
    print(prediction)
    print(conf)