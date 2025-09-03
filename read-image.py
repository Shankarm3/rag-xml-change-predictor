import google.generativeai as genai

genai.configure(api_key="AIzaSyCHDL4_qL_8fMG8aZM9co-1lP1ACYYXHqE")

model = genai.GenerativeModel("gemini-1.5-flash")

with open("D:/Shankar/image-testing/Capture001.png", "rb") as f:
    img_data = f.read()

response = model.generate_content([
    {"mime_type": "image/jpeg", "data": img_data},
    {"text": "Describe this image in detail."}
])

print(response.text)
