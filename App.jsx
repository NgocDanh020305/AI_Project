import React, { useState, useRef } from 'react';
import CanvasDraw from "react-canvas-draw";
import axios from 'axios';
import { Upload, PenTool, Eraser, CheckCircle, Copy, RefreshCw } from 'lucide-react';

function App() {
  const [mode, setMode] = useState('draw'); // 'draw' hoặc 'upload'
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultText, setResultText] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Xử lý khi chọn file upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResultText("");
    }
  };

  // Chuyển đổi Canvas thành Blob
  const getCanvasBlob = async () => {
    if (!canvasRef.current) return null;
    // Lấy dataURL từ canvas
    // Lưu ý: react-canvas-draw cần lấy canvasContainer để xuất ảnh đúng
    const canvas = canvasRef.current.canvasContainer.children[1];
    const dataUrl = canvas.toDataURL("image/png");

    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return blob;
  };

  // Gửi dữ liệu lên Backend
  const handleRecognize = async () => {
    setIsLoading(true);
    setResultText("");

    const formData = new FormData();

    try {
      if (mode === 'upload') {
        if (!imageFile) {
            alert("Vui lòng chọn ảnh trước!");
            setIsLoading(false);
            return;
        }
        formData.append('file', imageFile);
      } else {
        const blob = await getCanvasBlob();
        formData.append('file', blob, "drawing.png");
      }

      // Gọi API (Lưu ý đổi port nếu backend khác 8000)
      const response = await axios.post('http://localhost:8000/recognize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResultText(response.data.text || "Không nhận dạng được văn bản.");
    } catch (error) {
      console.error(error);
      setResultText("Lỗi kết nối tới Server hoặc lỗi xử lý.");
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(resultText);
    alert("Đã sao chép!");
  };

  const clearCanvas = () => {
    if (canvasRef.current) canvasRef.current.clear();
    setResultText("");
  };

  return (
    <div className="container">
      <header style={{textAlign: 'center', marginBottom: '30px'}}>
        <h1 style={{fontSize: '2.5rem', color: '#1f2937'}}>✍️ Handwritten Recognition AI</h1>
        <p style={{color: '#6b7280'}}>Nhận dạng chữ viết tay từ ảnh hoặc nét vẽ</p>
      </header>

      {/* Tab chuyển đổi chế độ */}
      <div style={{display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '20px'}}>
        <button
          className={`btn ${mode === 'draw' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('draw')}
        >
          <PenTool size={18} style={{marginRight: 5, verticalAlign: 'middle'}}/> Vẽ trực tiếp
        </button>
        <button
          className={`btn ${mode === 'upload' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('upload')}
        >
          <Upload size={18} style={{marginRight: 5, verticalAlign: 'middle'}}/> Tải ảnh lên
        </button>
      </div>

      <div style={{display: 'flex', gap: '20px', flexWrap: 'wrap'}}>

        {/* Cột Trái: Input */}
        <div style={{flex: 1, minWidth: '300px', background: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'}}>
          <h3 style={{marginTop: 0}}>Khu vực nhập liệu</h3>

          {mode === 'draw' ? (
            <div style={{border: '2px dashed #e5e7eb', borderRadius: '8px', overflow: 'hidden', display: 'flex', justifyContent: 'center'}}>
              <CanvasDraw
                ref={canvasRef}
                brushColor="#000"
                brushRadius={3}
                lazyRadius={0}
                canvasWidth={400}
                canvasHeight={300}
                gridColor="transparent"
              />
            </div>
          ) : (
            <div
              style={{height: '300px', border: '2px dashed #e5e7eb', borderRadius: '8px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'pointer'}}
              onClick={() => fileInputRef.current.click()}
            >
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" style={{maxWidth: '100%', maxHeight: '100%', objectFit: 'contain'}} />
              ) : (
                <>
                  <Upload size={48} color="#9ca3af"/>
                  <p style={{color: '#9ca3af'}}>Nhấn để chọn ảnh</p>
                </>
              )}
              <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{display: 'none'}} accept="image/*"/>
            </div>
          )}

          <div style={{marginTop: '15px', display: 'flex', gap: '10px'}}>
             {mode === 'draw' && (
                <button className="btn btn-secondary" onClick={clearCanvas}>
                  <Eraser size={18}/> Xóa
                </button>
             )}
             <button className="btn btn-primary" style={{flex: 1}} onClick={handleRecognize} disabled={isLoading}>
                {isLoading ? <RefreshCw className="spin" size={18}/> : <CheckCircle size={18} style={{marginRight: 5}}/>}
                {isLoading ? "Đang xử lý..." : "Nhận dạng ngay"}
             </button>
          </div>
        </div>

        {/* Cột Phải: Kết quả */}
        <div style={{flex: 1, minWidth: '300px', background: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'}}>
          <h3 style={{marginTop: 0}}>Kết quả văn bản</h3>
          <div style={{
            background: '#f9fafb',
            height: '300px',
            padding: '15px',
            borderRadius: '8px',
            border: '1px solid #e5e7eb',
            whiteSpace: 'pre-wrap',
            fontFamily: 'monospace',
            fontSize: '1.1rem',
            overflowY: 'auto'
          }}>
            {resultText || <span style={{color: '#9ca3af', fontStyle: 'italic'}}>Kết quả sẽ hiện ở đây...</span>}
          </div>

          {resultText && (
            <div style={{marginTop: '15px', textAlign: 'right'}}>
              <button className="btn btn-secondary" onClick={copyToClipboard}>
                <Copy size={18} style={{marginRight: 5}}/> Sao chép
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;