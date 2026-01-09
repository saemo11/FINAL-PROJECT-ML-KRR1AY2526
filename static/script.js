document.getElementById("predictForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())  // we will return JSON from Flask
    .then(data => {
        if (data.error) {
            document.getElementById("predictionOutput").innerHTML = `<span style="color:red;">${data.error}</span>`;
        } else {
            document.getElementById("predictionOutput").innerHTML = `<strong>Prediction:</strong> ${data.prediction} °${data.unit}`;
        }
    })
    .catch(err => {
        console.error("Prediction error:", err);
    });
});
