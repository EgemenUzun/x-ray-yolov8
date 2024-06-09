let startButton;
let stopButton;
let original_image;
let original_card;
let edited_image;
let edited_card;

document.addEventListener("DOMContentLoaded", function () {
	formatDate();
	startTimer();
	startButton = document.getElementById("baslat");
	stopButton = document.getElementById("bitir");
	original_card = document.getElementById("original-card");
	original_image = document.getElementById("original-image");
	edited_card = document.getElementById("edited-card");
	edited_image = document.getElementById("edited-image");
});
function formatDate() {
	let date = new Date().toLocaleString("en-GB", {
		hour12: false,
	});

	const header_date = document.getElementById("date");
	header_date.innerText = date;

	setTimeout(formatDate, 1000);
}

function startTimer() {
	// Get the current time in milliseconds
	let startTime = Date.now();

	// Set the interval to update the timer every second
	const intervalID = setInterval(function () {
		// Calculate the time elapsed since the timer started
		const elapsedTime = Date.now() - startTime;

		// Calculate the minutes and seconds
		const minutes = Math.floor(elapsedTime / 60000);
		const seconds = Math.floor(elapsedTime / 1000) % 60;

		// Update the timer display
		document.getElementById("timer").innerHTML =
			(minutes < 10 ? "0" + minutes : minutes) +
			":" +
			(seconds < 10 ? "0" + seconds : seconds);
	}, 1000);
}

const startWebcam = () => {
	original_image.src = originalUrl; // Use the videoUrl variable here
	edited_image.src = editedUrl;
};

const stopWebcam = () => {
	original_image.src = defaultImageUrl; // Use the videoUrl variable here
	edited_image.src = defaultImageUrl;
};
