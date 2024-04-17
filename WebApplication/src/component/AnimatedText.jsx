import React, { useEffect } from 'react';
import Anime from 'animejs'; // Import Anime.js
import "./AnimatedText.css"
const AnimatedText = ({ text }) => {
  useEffect(() => {
    const textWrapper = document.querySelector('.animated-text');
    textWrapper.innerHTML = textWrapper.textContent.replace(/\S/g, "<span class='letter'>$&</span>");

    Anime.timeline({ loop: true })
      .add({
        targets: '.animated-text .letter',
        translateY: [100, 0], // Upward movement
        translateZ: 0,
        opacity: [0, 1],
        easing: "easeOutExpo",
        duration: 1200,
        delay: (el, i) => 600 + 40 * i
      })
      .add({
        targets: '.animated-text .letter',
        translateY: [0, -100], // Downward movement
        opacity: [1, 0],
        easing: "easeInExpo",
        duration: 1600,
        delay: (el, i) => 400 + 60 * i
      });
  }, []); // Empty dependency array to run effect only once

  return (
    <h2 className="animated-text">{text}</h2> // Use the provided text
  );
};

export default AnimatedText;
