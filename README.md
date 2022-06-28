# Face Recognition with FaceNet (InceptionResnetV1) in Pytorch
![FR](assets/FR.png)


## Introduction
This is a mini-project of implementing Face Recognition using FaceNet in Pytorch on my internship at Widya Robotics

## Tutorial

Clone the project

```bash
git clone https://github.com/zogojogo/face-recognition-wii.git
```

Go to the project directory

```bash
cd face-recognition-wii
```

Download Dependencies
```bash
pip install -r requirements.txt
```

Download the Face Recognition model in this link and then put on the ./models/ folder
```http
https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt
```

Start API service

```
app.py --path <enrollment_path>
```
I made this program customizable so you can put your own enrollment path with face images inside to perform face recognition. The structure of the path is should be like this.

```bash
├── id1
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── id2
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── id3
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```
  
## API Reference

Service: http://your-ip-address:8080

#### POST image

```http
  POST /predict_identification
```
Content-Type: multipart/form-data
| Name    | Type   | Description                                         |
| :------ | :----- | :-------------------------------------------------- |
| `image` | `file` | **Required**. `image/jpeg` or `image/png` MIME Type |

```http
  POST /predict_verification
```
Content-Type: multipart/form-data
| Name    | Type   | Description                                         |
| :------ | :----- | :-------------------------------------------------- |
| `image` | `file` | **Required**. `image/jpeg` or `image/png` MIME Type |
| `image_2` | `file` | **Required**. `image/jpeg` or `image/png` MIME Type |

## Result Example (Identification)

**Input:**<br>
![input1](assets/0001_0.jpg)

**Output:**<br>
```python
{
  "filename": "0001_0.jpg",
  "contentype": "image/jpeg",
  "predicted id": "25",
  "confidence": "0.8785694285637182",
  "inference time": "0.05906987190246582"
}
```

---
## Result Example (Verification)

**Input:**<br>
![input2](assets/149358_2873934627303_944177111_n_0.jpg)

![input3](assets/1430066_0.jpg)

**Output:**<br>
```python
{
  "prediction": "Same Person!",
  "confidence": "0.8526605431392245",
  "inference time": "0.06154894828796387"
}
```