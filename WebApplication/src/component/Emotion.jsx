import React, {useState} from 'react';
import { SquareX } from 'lucide-react';
import "../component/Emotion.css";
import { CopyToClipboard } from 'react-copy-to-clipboard';
function Emotion({ result,onClose}) {
  console.log(result);
  const [textToCopy, setTextToCopy] = useState(''); // The text you want to copy
  const [copyStatus, setCopyStatus] = useState(false);
  const onCopyText = () => {
    setCopyStatus(true);
    setTimeout(() => setCopyStatus(false), 2000); // Reset status after 2 seconds
  };


// const textContainer = document.getElementById('text-container');
// const copyButton = document.getElementById('copy-button');

// if (textContainer && copyButton) { // Check if both elements exist before adding event listener
//   copyButton.addEventListener('click', () => {
//     if (textContainer &&textContainer.textContent) { // Double-check textContainer existence within click handler
//       const textToCopy = textContainer.textContent;

//       navigator.clipboard.writeText(textToCopy)
//         .then(() => {
//           console.log('Text copied to clipboard!');
//           // Optional: Show a success message to the user (e.g., using an alert or a UI update)
//         })
//         .catch(err => {
//           console.error('Failed to copy text:', err);
//           // Optional: Show an error message to the user (e.g., using an alert or a UI update)
//         });
//     } else {
//       console.error("Element with ID 'text-container' not found!");
//       // Optional: Handle missing element gracefully (e.g., disable button, provide informative message)
//     }
//   });
// } else {
//   console.error("Missing elements: 'text-container' or 'copy-button'");
//   // Optional: Handle missing elements globally (e.g., prevent script execution, display a message)
// }

//   // Conditional rendering based on result length

function getEmoji(emotion) {
  const emojis = {
    Happy: '😀',
    Sad: '😢',
    Angry: '😡',
    Fear: '😨',
    Neutral: '😐',
    Surprise: '😮'
  };
  return emojis[emotion] || null; // Return null if emotion is not found
}

const [bttnText, setBttnText] = useState("COPY CODE");
const emoji = getEmoji(result[0]?.label);

const copyCode = async () => {
  try {
    await navigator.clipboard.writeText(emoji || "Emotion result not available"); // Use label or fallback text
    setBttnText("COPIED");
    setCopyStatus(true);
    setTimeout(() => {
      setBttnText("COPY CODE");
      setCopyStatus(false);
    }, 3000);
  } catch (err) {
    console.log(err.message);
  }
};
  return (
    <div className='popup'>
      {result.length === 0 ? (
        <div>
            <div  className="close"><button  onClick={onClose}><SquareX color='#2f0be5' size={40}/></button></div>

            <div className="text-white">Unable to detect emotion.</div>
        </div>
        
      ) : (
        <div className='contain-result'>
            <div  className="close"><button  onClick={onClose}><SquareX color='#2f0be5' size={40}/></button></div>
            <div className='ls-2 fs-50 fw-6 result-component'>Your Result</div>
            <div className='result-emoji' id="text-container" 
        onChange={(e) => setTextToCopy(e.target.value)}>
                
                {result[0].label && ( // Check if label exists
              <span role="img" aria-label={result[0].label + " emoji"}>
                {emoji}
                
              </span>
            )}
              
            </div>
            {/* <div className='result-component'><button className='copy-button' id="copy-button">COPY</button></div> */}
            <div className='result-component'>
            <button className='copy-button fs-22' onClick={copyCode}>Copy</button>
              
            </div>
            {copyStatus && <div className='result-component fs-18'>Text copied to clipboard!</div>}
        </div>
      )}
    </div>
  );
}

export default Emotion;
