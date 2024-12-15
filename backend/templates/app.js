document.getElementById("checkButton").addEventListener("click", async () => {
  const userInput = document.getElementById("userInput").value;
  const resultDiv = document.getElementById("result");

  if (!userInput.trim()) {
    resultDiv.innerHTML = "<p style='color: red;'>Please enter text!</p>";
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: userInput }),
    });

    const data = await response.json();
    resultDiv.innerHTML = data.hate_speech
      ? "<p style='color: red;'>Hate speech detected!</p>"
      : "<p style='color: green;'>No hate speech detected.</p>";
  } catch (error) {
    resultDiv.innerHTML =
      "<p style='color: red;'>Error connecting to server.</p>";
    console.error(error);
  }
});
