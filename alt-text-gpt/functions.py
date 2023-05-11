# importing the requests library
import requests
import base64
import json
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv

load_dotenv()

# open-api statements
open_api_url = os.getenv('open_api_url')
open_api_key = os.getenv('open_api_key')

# every-pixel statements
ep_url = os.getenv('ep_url')
ep_username = os.getenv('ep_username')
ep_api_key = os.getenv('ep_api_key')

def get_keywords(img_url, ep_username, ep_api_key):
  auth = ep_username + ":" + ep_api_key
  encoded = base64.b64encode(auth.encode())

  auth_header = {"Authorization": "Basic " + str(encoded)}
  options = {'muteHttpExceptions': False, 'headers': auth_header}
  query = {'num_keywords': '20', 'url': img_url}

  # sending get request and saving the response as response object
  r = requests.get(url=ep_url,
                   headers=auth_header,
                   params=query,
                   auth=HTTPBasicAuth(ep_username, ep_api_key))

  keywordsStr = ''
  if (r.status_code == 200):
    data = r.json()
    print('> keyword data: ' + str(data))
    keywordsObjArray = data['keywords']

    for keyword in keywordsObjArray:
      keywordsStr += keyword['keyword'] + ", "
    #end-for()
  #end-if
  else:
    print('> failed to get keywords')
    print(r.json)
    return ''
  #end-else


  return keywordsStr
#end-get_key_words()


def call_gpt(img_url, open_api_key, keywords):
  #prompt = "Generate a descriptive ALT tag for an image based on the following keywords: " + keywords + "\n\nThe image is in the link: " + img_url + "\n\nDo not use the words vector, illustration, wallpaper, decoration or backdrop. Do not use qualitative adjectives. Write it in portuguese."

  prompt = "Generate a descriptive ALT tag for an image based on the following keywords: " + keywords + "\n\nDo not use the words vector, illustration, wallpaper, decoration or backdrop. Do not use qualitative adjectives. Write it in portuguese and add the hashtag #ParaTodosCegosVerem in the start of the alt text."

  data = {
    'model': "text-davinci-003",
    'prompt': prompt,
    'temperature': 0.7,
    'max_tokens': 256,
    'top_p': 1,
    'frequency_penalty': 0,
    'presence_penalty': 0,
    'best_of': 1,
    'stop': ["####"],
  }

  auth_header = {
    "Authorization": "Bearer " + open_api_key,
    'contentType': "application/json",
  }

  r = requests.post(url=open_api_url, headers=auth_header, json=data)
  print("text propmt: " + prompt)
  print("output json: " + str(r.json()))


#end-call_gpt()


def get_alt_text(img_url):
  # Exit function if no parameter is provided
  if (img_url == ""):
    print(">>> empty url. program ended.")
    return ""
  #end-if

  keywordString = get_keywords(img_url, ep_username, ep_api_key)

  print('> keywords found: ' + keywordString)

  if (keywordString == '' and False):
    print(">>> empty keywords. program ended.")
    return ""
  #end-if()

  call_gpt(img_url, open_api_key, keywordString)
