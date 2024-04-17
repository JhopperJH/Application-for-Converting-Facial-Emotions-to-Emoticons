import Webcam from "react-webcam";
import React, {useCallback,useState, useRef} from 'react';
import "./WebCam.css"
import { SquareX } from 'lucide-react';

function WebCam({onClose, onCapture}) {

  const retake = () => {
    setImgSrc(null);
  };

  const webcamRef = useRef(null);
  const [imgSrc, setImgSrc] = useState(""); 
  const [imageFile, setImageFile] = useState(null); // Stores File object (optional)
  const [oriUrl,setOriUrl] = useState("");
  const capture = useCallback(() => {
    const capturedImage = webcamRef.current.getScreenshot();
    setOriUrl(capturedImage);
    // Handle both WebP and potential future PNG cases
    if (capturedImage.startsWith('data:image/webp;base64,')) {
      
      // WebP Base64 Conversion (using fetch for compatibility)
      fetch(capturedImage)
      
        .then(response => response.blob())
        .then(blob => {
          const reader = new FileReader();
          reader.readAsDataURL(blob);
          reader.onloadend = () => {
            const convertedImage = reader.result; // Converted PNG data URL
            setImgSrc(convertedImage); // Update for preview

            // Optional: Create File object (if needed for upload)
            const filename = 'captured_image.png';
            const file = new File([blob], filename, { type: 'image/png' });
            setImageFile(file);
            console.log("test web cam0",file);

            // Call onCapture with both data URL and File object (if created)
            if (onCapture) {
              console.log("test from webcam origi url",capturedImage);
              console.log("from last stagewebcam",file);
              onCapture(capturedImage,file); // Pass both values

              //onCapture(convertedImage, file); // Pass both values
         
            }
          };
        })
        .catch(error => console.error('Error converting WebP image:', error));
    } else {
      // Handle potential future PNG data URLs directly
      setImgSrc(capturedImage); // Update for preview
      setImageFile(null); // Clear File object if not needed

      // Optional: Call onCapture with data URL (if PNG)
      if (onCapture) {
        onCapture(capturedImage, null); // Pass data URL, no File object
      }
    }
  }, [webcamRef]);

  // const capture = useCallback(() => {
  //   const imageSrc = webcamRef.current.getScreenshot();
  //   setImgSrc(imageSrc);
  //   // Call the onCapture prop function with the captured imageSrc
  //   if (onCapture) {
  //     onCapture(imageSrc);
  //   }
  // }, []);


  return (
    <div className="popup">
      <div className="contain-cam">
        <div  className="close">
          <button onClick={onClose}><SquareX color='#2f0be5' size={40}/></button>
        </div>
        <div className="cam-div">
          {imgSrc ? (
          <img src={imgSrc} alt="webcam" />
          ) : (
            <Webcam mirrored={true} ref={webcamRef} className='react-webcam'/>
          )}
        </div>
        <div className="">
          {imgSrc ? (
            <div  className="btn-container">
              <button className="re-btn fw-6 fs-22 flex" onClick={retake}> retake </button>
              <button className="snap-btn fw-6 fs-22 flex" onClick={onClose}>USE PHOTO</button>
          </div>
            
          ) : (
            <div  className="btn-container">
            <button className="cap-btn fw-6 fs-22 flex" onClick={capture}>Capture photo</button>
            </div>
          )}
      </div>
      </div>
      
      
    </div>
  )
}

export default WebCam