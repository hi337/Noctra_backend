const socket = io("http://localhost:3000");

socket.on("video-update", (data) => {
  document.getElementById("videoStream").src = "data:image/jpeg;base64," + data;
});

document.getElementById("toggleStream").addEventListener("click", () => {
  socket.emit("start-stream");
});
