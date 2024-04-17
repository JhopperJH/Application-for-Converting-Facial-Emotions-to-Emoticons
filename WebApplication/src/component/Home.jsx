import React,{useEffect, useState} from 'react'

import { Outlet } from 'react-router-dom'
import FirebaseUploadImage from '../FireBase/FirebaseUploadImage'
import AnimatedText from "./AnimatedText"
import "./Home.css"

const Home = () => {
 
  
  
  return (
    <div className='setBackground'>
      <div className='title section-title'>
        <AnimatedText text="Upload your photo!" />
      </div>
      <FirebaseUploadImage />

    </div>
  )
}

export default Home