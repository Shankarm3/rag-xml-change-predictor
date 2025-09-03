import google.generativeai as genai

genai.configure(api_key="AIzaSyCHDL4_qL_8fMG8aZM9co-1lP1ACYYXHqE")

response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
    "google.generativeai"
)

print(response.text)
