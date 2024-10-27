import { useState } from "react";

function App() {
  // Initialize form state
  const [formData, setFormData] = useState({
    text: "",
  });

  const [result, setResult] = useState(0);
  // Handle input changes
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        cors: "no-cors",
        body: JSON.stringify(formData),
      });
      if (response.ok) {
        const result = await response.json();
        setResult(result.prediction);
        console.log(result);
      } else {
        alert("Failed to send data!");
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };
  return (
    <div className={`h-screen w-screen  bg-slate-800  p-6`}>
      <h1 className="font-semibold font-sans text-4xl tracking-tighter text-white">
        Sentiment Classification
      </h1>
      <form className="mt-6" onSubmit={handleSubmit}>
        <input
          className="w-full mt-2 p-2 text-slate-800"
          type="text"
          name="text"
          onChange={handleChange}
        />
        <button
          className="mt-4 bg-white text-slate-800 font-semibold p-2 rounded shadow-md min-w-[10vw] hover:bg-slate-800 hover:text-white"
          type="submit"
        >
          Submit
        </button>
      </form>
      <div
        className={
          "py-5 text-7xl " +
          (result == "Positive" ? "text-green-400" : "text-red-400")
        }
      >
        {result}
      </div>
    </div>
  );
}

export default App;
