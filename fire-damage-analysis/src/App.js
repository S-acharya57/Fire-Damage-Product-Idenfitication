import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./styles.css";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const canvasRef = useRef(null);

  // selecting image
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
  };

  // bounding boxes
  const drawBoundingBoxes = (image, predictions) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.font = "16px Arial";
    // colors for bounding boxes
    const colors = [
      "rgba(255, 99, 71, 0.6)",
      "rgba(135, 206, 235, 0.6)",
      "rgba(144, 238, 144, 0.6)",
      "rgba(255, 215, 0, 0.6)",
      "rgba(255, 165, 0, 0.6)",
      "rgba(221, 160, 221, 0.6)",
    ];

    predictions.forEach((item, index) => {
      const color = colors[index % colors.length];
      const [xmin, ymin, xmax, ymax] = item.bbox;

      // bbbox
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

      // confidence and name
      ctx.fillStyle = color;
      ctx.fillText(
        `${item.category} (${(item.confidence * 100).toFixed(1)}%)`,
        xmin,
        ymin - 10
      );
    });
  };

  // Crop image
  const cropImage = (image, bbox) => {
    const [xmin, ymin, xmax, ymax] = bbox;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    // canvas to bb box size
    canvas.width = xmax - xmin;
    canvas.height = ymax - ymin;

    // cropped region image
    ctx.drawImage(
      image,
      xmin,
      ymin,
      canvas.width,
      canvas.height,
      0,
      0,
      canvas.width,
      canvas.height
    );

    return canvas.toDataURL(); // cropped image URL
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) {
      alert("Please select an image to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Error processing the image.");
    }
  };

  // Redraw if results are updated
  useEffect(() => {
    if (result && imagePreview) {
      const image = new Image();
      image.src = imagePreview;
      image.onload = () => drawBoundingBoxes(image, result.items);
    }
  }, [result, imagePreview]);

  // download CSV
  const downloadCSV = () => {
    if (!result) return;

    const headers = [
      "Category",
      "Predicted Brand",
      "Confidence",
      "Clip Confidence",
      "Bounding Box",
    ];

    const rows = result.items.map((item) => [
      item.category,
      item.predicted_brand,
      (item.confidence * 100).toFixed(1) + "%",
      (item.clip_confidence * 100).toFixed(1) + "%",
      item.bbox.join(", "),
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute("href", url);
      link.setAttribute("download", "analysis_results.csv");
      link.click();
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Fire Damage Product Identification</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Upload Image:
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </label>
        <button type="submit">Submit</button>
      </form>

      {imagePreview && (
        <div style={{ position: "relative", marginTop: "20px" }}>
          <h2>Uploaded Image</h2>
          <div style={{ position: "relative" }}>
            <img
              src={imagePreview}
              alt="Uploaded Preview"
              style={{
                maxWidth: "100%",
                height: "auto",
                display: "block",
                position: "relative",
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                pointerEvents: "none",
              }}
            />
          </div>
        </div>
      )}

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h2>Analysis Results</h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: "20px",
            }}
          >
            {result.items.map((item, index) => {
              const image = new Image();
              image.src = imagePreview;

              return (
                <div key={index} style={{ textAlign: "center" }}>
                  {/* image on left */}
                  <img
                    src={cropImage(image, item.bbox)}
                    alt={`Cropped ${item.category}`}
                    style={{
                      maxWidth: "150px",
                      height: "auto",
                      border: "1px solid #ccc",
                      borderRadius: "5px",
                      marginBottom: "10px",
                    }}
                  />
                  {/* analysis below image */}
                  <div>
                    <strong>Category:</strong> {item.category} <br />
                    <strong>Predicted Brand:</strong> {item.predicted_brand}{" "}
                    <br />
                    <strong>Confidence:</strong>{" "}
                    {(item.confidence * 100).toFixed(1)}% <br />
                    <strong>Clip Confidence:</strong>{" "}
                    {(item.clip_confidence * 100).toFixed(1)}% <br />
                    <strong>Bounding Box:</strong> {item.bbox.join(", ")} <br />
                  </div>
                </div>
              );
            })}
          </div>
          <button onClick={downloadCSV} style={{ marginTop: "20px" }}>
            Download CSV
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
