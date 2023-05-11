//Get API Keys from Settings sheet
const ss = SpreadsheetApp.getActiveSpreadsheet();
const sheet = ss.getSheetByName("Settings");
const EP_USERNAME = sheet.getRange('A2').getValue();
const EP_PASSWORD = sheet.getRange('B2').getValue();
const Openapikey = sheet.getRange('C2').getValue();
const RAPIDAPI_KEY = sheet.getRange('D2').getValue();
const INC_COLORS = sheet.getRange('E2').getValue();


// Return the colours in the image as RGB values using Regim API via RapidAPI
function getColors(imageurl){

  const sheet = SpreadsheetApp.getActiveSheet();

  //Grab Image 
  const imgURL = imageurl;
  const imgBlob = UrlFetchApp.fetch(imgURL).getBlob();

  //Prepare Request for RapidAPI
  var options = {
    method: "POST",
    headers: {
      'ContentType': 'multipart/form-data; boundary=---011000010111000001101001',
      'X-RapidAPI-Key': RAPIDAPI_KEY,
      'X-RapidAPI-Host': 'regim3.p.rapidapi.com'
    },
    payload: {
      file: imgBlob
    }
  };
  
  var url = encodeURI("https://regim3.p.rapidapi.com/1.1/?opts=colors")
  var res = UrlFetchApp.fetch(url, options);

 //Received a response, adding it to Spreadsheet
  if(res.getResponseCode() === 200){
    var respText = JSON.parse(res.getContentText());
   // Logger.log(respText);
    var colors = respText.data.colors.toString();
    return colors;
  }

}


// Get the image keywords from EveryPixel API

function getKeywords(varimage){

  // Exit function if no parameter is provided
  if (varimage === "") {
      return "";
  }

  var options = {
    'muteHttpExceptions': true,
    'headers': {
      Authorization: "Basic " + Utilities.base64Encode(EP_USERNAME + ":" + EP_PASSWORD)
    },
  };

  const url = 'https://api.everypixel.com/v1/keywords?num_keywords=10&url=' + varimage;
  var response = UrlFetchApp.fetch(url, options);
  Logger.log("KWs Raw Response: " + response);

  if(response.getResponseCode() === 200){
    var keywords = JSON.parse(response.getContentText()).keywords;
    var strKeywords = '';

//convert the response into a comma-separated single line 
    for(var i=0;i<keywords.length;i++){
      strKeywords += keywords[i].keyword + ', ';
    }
//Logger.log ("Keywords Formatted: " + strKeywords)
    return strKeywords;
  }
  return '';
}

// Call OpenAPI to get the color names from Hex Values

function getColNames(imageurl){

var varRGB = getColors(imageurl)

  var data = {
 "model": "text-davinci-003",
  "prompt": "Describe the standard colour names using British English spelling and represented by the hex codes. \n\nHex Codes: #dce8c6,#353431,#b76928\nColour descriptions: Light mint green, dark grey, dark orange\n###\n\nHex Codes: " + varRGB + "\nColour descriptions:",
  "temperature": 0.03,
  "max_tokens": 566,
  "top_p": 1,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "stop": ["###"]
    
  };

  var options = {
    'method' : 'post',
    'contentType': 'application/json',
    'payload' : JSON.stringify(data),
    'headers': {
      Authorization: 'Bearer ' + Openapikey,
    },
  };

  var response = UrlFetchApp.fetch(
    'https://api.openai.com/v1/completions',
    options,
  );

var result = JSON.parse(response.getContentText())['choices'][0]['text']

// trim the response
var result = result.replace(/^\s+|\s+$/gm, '');
//Logger.log("Colour Names: "+ result);
return result
}



// Call OpenAPI to convert image keywords and colours to ALT tag

function getALT(varimage, varwords){

  // Exit function if no parameter is provided
  if (varimage === "") {
      return "";
  }

var varkws = getKeywords(varimage)
var varwordsinc = varwords

// Add the required words if they exist
if (varwordsinc != "") {
  varwordsinc = "The ALT MUST include the words, " + varwordsinc;
} else {
  varwordsinc = "";
}


if (INC_COLORS == "YES") {
  var ColNames = " and the colours " + getColNames(varimage);
} else {
  var ColNames = "";
}

//var varprompt = "Generate a descriptive ALT tag of no more than 16 words for an image based on the following keywords: " + varkws + ColNames + "\n\nDo not use the words vector, illustration, wallpaper, decoration or backdrop.\n\nALT Tag:"

//Logger.log (varprompt) 

  var data = {
   "model": "text-davinci-003",
    "prompt": "Generate a descriptive ALT tag of no more than 16 words for an image based on the following keywords: " + varkws + ColNames + ".\n\n" + varwordsinc + ".\n\nDo not use the words vector, illustration, wallpaper, decoration or backdrop.\n\nALT Tag:",  

    "temperature": 0.7,
    "max_tokens": 256,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "best_of": 1,
    "stop": ["####"]
    
  };

  var options = {
    'method' : 'post',
    'contentType': 'application/json',
    'payload' : JSON.stringify(data),
    'headers': {
      Authorization: 'Bearer ' + Openapikey,
    },
  };

  var response = UrlFetchApp.fetch(
    'https://api.openai.com/v1/completions',
    options,
  );

var result = JSON.parse(response.getContentText())['choices'][0]['text']

// trim the response
var result = result.replace(/^\s+|\s+$/gm, '');
var result = result.replace(/\"/g, "");
//Logger.log("GET ALT Tag: "+ result);
return result
}