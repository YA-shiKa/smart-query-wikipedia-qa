const form = document.getElementById("ragForm");
const topicInput = document.getElementById("topic");
const questionInput = document.getElementById("question");
const responseDiv = document.getElementById("response");
const loadingSpinner = document.getElementById("loading");
const toggleDark = document.getElementById("toggleDark");
const body = document.body;
const container = document.querySelector("div.w-full");

// Apply saved dark mode
if (localStorage.getItem("dark") === "true") {
  enableDarkMode();
}

toggleDark.addEventListener("click", () => {
  if (body.classList.contains("dark")) {
    disableDarkMode();
  } else {
    enableDarkMode();
  }
});

function enableDarkMode() {
  body.classList.add("dark");
  body.classList.remove("bg-gray-100", "text-gray-800");
  body.classList.add("bg-gray-900", "text-gray-200");
  container.classList.remove("bg-white");
  container.classList.add("bg-gray-800");
  localStorage.setItem("dark", "true");
}

function disableDarkMode() {
  body.classList.remove("dark");
  body.classList.remove("bg-gray-900", "text-gray-200");
  body.classList.add("bg-gray-100", "text-gray-800");
  container.classList.remove("bg-gray-800");
  container.classList.add("bg-white");
  localStorage.setItem("dark", "false");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  responseDiv.innerHTML = "";
  loadingSpinner.classList.remove("hidden");

  const topic = topicInput.value;
  const question = questionInput.value;

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, question })
    });
    const data = await res.json();
    loadingSpinner.classList.add("hidden");

    if (data.error) {
      responseDiv.innerHTML = `<div class="text-red-600">${data.error}</div>`;
    } else {
      const answerBlock = `
        <div class="p-4 rounded-lg border dark:border-gray-600">
          <div><strong>Topic:</strong> ${topic}</div>
          <div><strong>Question:</strong> ${question}</div>
          <div><strong>Answer:</strong> ${data.answer}</div>
          <details class="mt-2">
            <summary class="cursor-pointer text-indigo-600">Show retrieved context</summary>
            <ul class="list-disc pl-5 mt-1">
              ${data.retrieved_chunks.map(c => `<li>${c}</li>`).join('')}
            </ul>
          </details>
        </div>
      `;
      responseDiv.innerHTML = answerBlock + responseDiv.innerHTML;
    }
  } catch (err) {
    loadingSpinner.classList.add("hidden");
    responseDiv.innerHTML = `<div class="text-red-600">Error: ${err.message}</div>`;
  }
});
