import React, { useState } from 'react';
import './VideoUpload.css';

function VideoUpload({ onVideoUpload, loading }) {
  const [videoPreview, setVideoPreview] = useState(null);

  const handleVideoChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const videoFile = e.target.files[0];
      setVideoPreview(URL.createObjectURL(videoFile));
      onVideoUpload(videoFile);
    }
  };

  return (
    <div className="upload-section">
      <h2>Upload a Video</h2>
      <input type="file" accept="video/*" onChange={handleVideoChange} disabled={loading} />
      {videoPreview && (
        <div className="video-preview">
          <video controls>
            <source src={videoPreview} type="video/mp4" />
            Your browser does not support HTML video.
          </video>
        </div>
      )}
    </div>
  );
}

export default VideoUpload;