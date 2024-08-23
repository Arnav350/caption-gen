import React, { useState, ChangeEvent } from "react";
import "./App.css";

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [caption, setCaption] = useState<string>("");

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setCaption(data.caption);
  };

  return (
    <div className="App">
      <h1>Image Caption Generator</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Generate Caption</button>
      {caption && <p>Caption: {caption}</p>}
    </div>
  );
};

export default App;
