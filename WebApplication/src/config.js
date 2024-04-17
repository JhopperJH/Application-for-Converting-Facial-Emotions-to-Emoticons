// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import {getStorage} from "firebase/storage";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAWNnr-SAWP0UI8Ov06aURYiNGRbDjXYRA",
  authDomain: "ml-project-1130f.firebaseapp.com",
  projectId: "ml-project-1130f",
  storageBucket: "ml-project-1130f.appspot.com",
  messagingSenderId: "174330516021",
  appId: "1:174330516021:web:487b7f2a0421dfc917577c",
  measurementId: "G-TGBNEMHRWC"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const imageDb = getStorage(app)