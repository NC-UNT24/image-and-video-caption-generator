// CaptionDisplay.js
import React from 'react';
import './CaptionDisplay.css';
import PropTypes from 'prop-types';

function CaptionDisplay({ caption, error }) {
  /**
   * Determines whether a caption exists.
   * - For arrays: checks if the array has at least one element.
   * - For strings: checks if the string is not empty or just whitespace.
   * - For other types: returns false.
   *
   * @returns {boolean} - True if a caption exists, false otherwise.
   */
  const hasCaption = () => {
    if (Array.isArray(caption)) {
      return caption.length > 0;
    } else if (typeof caption === 'string') {
      return caption.trim() !== '';
    }
    return false;
  };

  return (
    <div className="caption-display">
      {error ? (
        <>
          <h2 className="error">Error:</h2>
          <p className="error">{error}</p>
        </>
      ) : hasCaption() ? (
        Array.isArray(caption) ? (
          <>
            <h2>Generated Captions:</h2>
            <ul>
              {caption.map((item, index) => (
                <li key={index}>
                  Frame {item.frame}: {item.caption}
                </li>
              ))}
            </ul>
          </>
        ) : (
          <>
            <h2>Generated Caption:</h2>
            <p>{caption}</p>
          </>
        )
      ) : (
        <p>No caption generated yet.</p>
      )}
    </div>
  );
}

CaptionDisplay.propTypes = {
  caption: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.arrayOf(
      PropTypes.shape({
        frame: PropTypes.number.isRequired,
        caption: PropTypes.string.isRequired,
      })
    ),
  ]),
  error: PropTypes.string,
};

CaptionDisplay.defaultProps = {
  caption: null,
  error: null,
};

export default CaptionDisplay;