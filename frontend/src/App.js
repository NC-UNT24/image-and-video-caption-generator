import { ClipLoader } from 'react-spinners';
import './App.css';
import React, { useState } from 'react';
import Header from './components/Header';
import LanguageSelector from './components/LanguageSelector';
import ImageUpload from './components/ImageUpload';
import VideoUpload from './components/VideoUpload';
import CaptionDisplay from './components/CaptionDisplay';
import './styles/global.css'; // Ensure global styles are imported

function App() {
  const [error, setError] = useState('');
  
  // Separate states for image and video captions
  const [imageCaption, setImageCaption] = useState('');
  const [videoCaptions, setVideoCaptions] = useState([]); 
  
  const [language, setLanguage] = useState('en'); 
  const [loading, setLoading] = useState(false);

  const handleLanguageChange = (lang) => {
    setLanguage(lang);
  };

  const handleImageUpload = async (imageFile) => {
    setLoading(true);
    setError('');
    setImageCaption('');
    // Do not clear videoCaptions here

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('language', language);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/generate-image-caption', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setImageCaption(data.caption); 
      } else {
        setError(data.error || 'An error occurred while generating the caption.');
      }
    } catch (err) {
      setError('An error occurred while connecting to the server.');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async (videoFile) => {
    setLoading(true);
    setError('');
    setVideoCaptions([]);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('language', language);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/generate-video-caption', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setVideoCaptions(data.captions); 
      } else {
        setError(data.error || 'An error occurred while generating captions.');
      }
    } catch (err) {
      setError('An error occurred while connecting to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />

      {/* Centered language selector */}
      <div className="language-selector-container">
        <LanguageSelector selectedLanguage={language} onLanguageChange={handleLanguageChange} />
      </div>

      {loading && (
        <div className="loader">
          <ClipLoader color="#FFFFFF" loading={loading} size={50} />
        </div>
      )}

      <div className="layout-container">
        {/* Image Column */}
        <div className="image-column">
          <div className="image-upload-section">
            <ImageUpload onImageUpload={handleImageUpload} loading={loading} />
          </div>
          <div className="image-caption-section">
            <h2>Image Caption</h2>
            <CaptionDisplay caption={imageCaption} error={error} />
          </div>
        </div>

        {/* Video Column */}
        <div className="video-column">
          <div className="video-upload-section">
            <VideoUpload onVideoUpload={handleVideoUpload} loading={loading} />
          </div>
          <div className="video-caption-section">
            <h2>Video Captions</h2>
            <CaptionDisplay caption={videoCaptions} error={error} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;