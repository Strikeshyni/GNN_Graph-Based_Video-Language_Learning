import sng_parser

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query(img_count)
img_caption = output[0]['generated_text']
graph = sng_parser.parse(img_caption)