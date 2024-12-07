import React, { useState } from 'react';
import './ImageUpload.css';

function ImageUpload({ onImageUpload, loading }) {
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const imageFile = e.target.files[0];
      setImagePreview(URL.createObjectURL(imageFile));
      onImageUpload(imageFile);
    }
  };

  return (
    <div className="upload-section">
      <h2>Upload an Image</h2>
      <input type="file" accept="image/*" onChange={handleImageChange} disabled={loading}/>
      {imagePreview && (
        <div className="image-preview">
          <img src={imagePreview} alt="Uploaded" />
        </div>
      )}
    </div>
  );
}

export default ImageUpload;