const https = require("https");
const fs = require("fs");
const path = require("path");

const options = {
  key: fs.readFileSync("./10.100.103.0+1-key.pem"),
  cert: fs.readFileSync("./10.100.103.0+1.pem"),
};

const server = https.createServer(options, (req, res) => {
  let filePath = "." + (req.url === "/" ? "/index.html" : req.url);
  const ext = path.extname(filePath);

  const mimeTypes = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
  };

  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
    } else {
      res.writeHead(200, { "Content-Type": mimeTypes[ext] || "text/plain" });
      res.end(content);
    }
  });
});

server.listen(3000, "0.0.0.0", () => {
  console.log("HTTPS server running at https://10.100.103.0:3000");
});
