import React, { useEffect, useRef, useState } from "react";
import { imageDb } from '../config';
import { getDownloadURL, listAll, ref, uploadBytes } from "firebase/storage";
import { v4 } from "uuid";
import "./FirbaseUploadImage.css";
import takeSelfie from "../assets/takeSelfie.json";
import Lottie from "lottie-react";
import axios from "axios";
import WebCam from "../WebCam/WebCam";

import Emotion from "../component/Emotion";


function FirebaseUploadImage() {
 
  const [img, setImg] = useState(null); // Store the selected image as null initially
  const [imgUrl, setImgUrl] = useState(''); // Store the URL of the selected image
  const [showAnimation, setShowAnimation] = useState(true);
  const [showCam, setShowcam]=useState(false);
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [showEmo, setShowEmo] = useState(false);

  const button = document.querySelector('.button');
  const addEventListener=(e)=>{
    e.preventDefault();
    button.classList.add("animate");

    setTimeout(()=>{

    },600)
  }

  const handleImage = (event) => {
    const file = event.target.files[0];
    // Update image state (assuming setImg is for image preview)
    console.log("normal file:",file)
    setImg(file);
    if (!file) {
      console.warn('No file selected. Cancelling image preview.');
      return; // Early return if no file is selected
    }
    // Check if image preview is desired (optional)
    if (setImgUrl) {
      const reader = new FileReader();
      reader.onload = (e) => setImgUrl(e.target.result);
      reader.readAsDataURL(file);
    }
      //console.log("file", file);
      setInput(file);
  };

  const handleCapImage = (event) => {
    
    const file = event.target.files[0];
    // Update image state (assuming setImg is for image preview)
    console.log("cap in handlecap file:",file)
    setImg(file);
    if (!file) {
      console.warn('No file selected. Cancelling image preview.');
      return; // Early return if no file is selected
    }
    // Check if image preview is desired (optional)
    if (setImgUrl) {
      const reader = new FileReader();
      reader.onload = (e) => setImgUrl(e.target.result);
      reader.readAsDataURL(file);
    }
      //console.log("file", file);
      setInput(file);
  };

  const handleCamera = async () => {
    setShowcam(!showCam);
  }
  
  const handleApi = async () => {

   

    console.log("Check input in handle Api:",input)
    if (!input) { // Check if image is selected before upload
      console.error('Please select an image to upload.');
      return; // Early return to prevent unnecessary processing
    }
    const formData = new FormData();
    formData.append('image', input);
    console.log("formData:",formData.has('image'));
    
    try {
      const response = await axios.post('/predict', formData);
      console.log("Fuck what?");
      setOutput(response.data);
      setShowEmo(true);
      // Assuming response.data contains predicted results

    } catch (error) {
      console.error('Error uploading image:', error);
      
    }
  };

  const handleClick = async () => {
    if (img) {
      const imgRef = ref(imageDb, `files/${v4()}`);
      await uploadBytes(imgRef, img); // Use async/await for clarity
      const url = await getDownloadURL(imgRef);
      setImgUrl(url);
      setImg(null); // Clear the selected image after upload
      setShowAnimation(false); // Hide animation after upload
    }
  };
 
  
  
  
  const handleCapture = (oriUrl,capturedImageSrc) => {
    console.log('Captured image:', capturedImageSrc);
    setImgUrl(oriUrl);

    setImg(capturedImageSrc); // For image preview (optional)
    setInput(capturedImageSrc); // For upload and processing
    
    console.log("chuck file here",input);
  };


  useEffect(() => {
    // Show animation initially and hide on first upload
    setShowAnimation(!img); // Only show when img is null (no image)
  }, [img]); // Dependency array: Re-run only on img change (upload)

  return (
    <div className="containbox">
      <div className="containbox">
        <div className="lottie-container">
          {showAnimation && ( // Conditionally render the animation
            <Lottie animationData={takeSelfie} />
          )}
          {
            imgUrl && (  // Only render the image if imgUrl is not empty
              <div>
                <img src={imgUrl}  alt="Unable to upload image"/>
                <br />
              </div>
            )
          }
        </div>
      </div>
      <div className="containbox">

          <div className="center-ja">
            <button className="choose-btn fw-6 fs-22 flex" >
            <label for="file-upload" class="choosefile">
        Choose File
    </label>
              <input className=''  id="file-upload" type="file" onChange={handleImage} />

            </button>
          </div>
          <div  className="upload-btn">
            <button className="detect-btn  fw-6 fs-22 flex" 
              onClick={handleApi}> DETECT PIC</button>
            <button className="cam-btn fw-6 fs-22 flex" onClick={handleCamera}>USE CAMERA !</button>
          </div>
          <div ></div>
          {showEmo &&(
            <div className="containbox fs-26 overlay">
              <Emotion result={output} onClose={()=>setShowEmo(false)}/>
            </div>
          )}
          {showCam &&(
            <WebCam onCapture={handleCapture} onClose={()=>setShowcam(false) }/>
            
          )}
      </div>
      
      
    </div>
  );
}

export default FirebaseUploadImage;


/*import React, { useEffect, useState } from "react";
import { imageDb } from './config';
import { getDownloadURL, listAll, ref, uploadBytes } from "firebase/storage";
import { v4 } from "uuid";
import "./FirbaseUploadImage.css"


function FirebaseUploadImage() {
  const [img, setImg] = useState(null); // Store the selected image as null initially
  const [imgUrl, setImgUrl] = useState(''); // Store the URL of the single displayed image

  const handleClick = async () => {
    if (img) {
      const imgRef = ref(imageDb, `files/${v4()}`);
      await uploadBytes(imgRef, img); // Use async/await for clarity
      const url = await getDownloadURL(imgRef);
      setImgUrl(url);
      setImg(null); // Clear the selected image after upload
    }
  }

  useEffect(() => {
    listAll(ref(imageDb, "files")).then(imgs => {
      console.log(imgs);
      if (imgs.items.length > 0) {
        getDownloadURL(imgs.items[0]).then(url => {
          setImgUrl(url);
        })
      }
    })
  }, []) // Empty dependency array to fetch only on initial render


  return (
    <div className="containbox">
        <div className="upload-btn">
            <input type="file" onChange={(e) => setImg(e.target.files[0])} />
            <button className="btn" onClick={handleClick}>Upload</button>
        </div>
        <br/>
        {
            imgUrl && (  // Only render the image if imgUrl is not empty
            <div>
                <img src={imgUrl} height="350px" width="200px" />
                <br/>
            </div>
            )
        }
    </div>
  )
}

export default FirebaseUploadImage;
*/