// AI Summary: Contains all JavaScript functionality for the ACE project page
// Handles citation copying, image lightbox, back-to-top navigation,
// and collapsible sections with accessibility features

document.addEventListener("DOMContentLoaded", function () {
  // Citation copy functionality
  const citations = document.querySelectorAll(".citation");

  citations.forEach(function (citation, index) {
    // Create the button with Lucide clipboard icon
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.setAttribute("aria-label", "Copy to clipboard");

    // Create SVG icon (Lucide clipboard icon)
    copyBtn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;

    // Add the button to the citation
    citation.appendChild(copyBtn);

    // Add click event to copy the content
    copyBtn.addEventListener("click", function () {
      // Get the text content, making sure to ignore the SVG content
      const textNodes = Array.from(citation.childNodes)
        .filter(
          (node) =>
            node.nodeType === Node.TEXT_NODE ||
            (node.nodeType === Node.ELEMENT_NODE &&
              node.tagName !== "BUTTON")
        )
        .map((node) => node.textContent);

      const text = textNodes.join("").trim();

      // Copy to clipboard
      navigator.clipboard
        .writeText(text)
        .then(function () {
          // Enhanced visual feedback with checkmark icon
          copyBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          `;
          copyBtn.classList.add("copied");
          setTimeout(function () {
            copyBtn.innerHTML = `
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            `;
            copyBtn.classList.remove("copied");
          }, 2000);
        })
        .catch(function (err) {
          console.error("Could not copy text: ", err);
          copyBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          `;
          setTimeout(function () {
            copyBtn.innerHTML = `
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            `;
          }, 2000);
        });
    });
  });

  // Image lightbox functionality
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxClose = document.querySelector(".lightbox-close");
  const images = document.querySelectorAll("img:not(#lightbox-img)");

  // Setup image click listeners
  images.forEach(function (img) {
    img.addEventListener("click", function () {
      lightboxImg.src = this.src;
      lightboxImg.alt = this.alt;
      lightbox.classList.add("active");
      document.body.style.overflow = "hidden"; // Prevent scrolling when lightbox is open
      lightbox.setAttribute("aria-hidden", "false");
    });
  });

  // Close lightbox when clicking the close button
  lightboxClose.addEventListener("click", closeLightbox);

  // Close lightbox when clicking outside the image
  lightbox.addEventListener("click", function (e) {
    if (e.target === lightbox) {
      closeLightbox();
    }
  });

  // Close lightbox when pressing Escape key
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && lightbox.classList.contains("active")) {
      closeLightbox();
    }
  });

  function closeLightbox() {
    lightbox.classList.remove("active");
    document.body.style.overflow = ""; // Restore scrolling
    setTimeout(function () {
      lightboxImg.src = ""; // Clear the source after transition
    }, 300);
    lightbox.setAttribute("aria-hidden", "true");
  }

  // Back to top button functionality
  const backToTopButton = document.getElementById("back-to-top");

  // Show/hide back to top button based on scroll position
  window.addEventListener("scroll", function () {
    if (window.pageYOffset > 300) {
      backToTopButton.classList.add("visible");
    } else {
      backToTopButton.classList.remove("visible");
    }
  });

  // Smooth scroll to top when clicking the button
  backToTopButton.addEventListener("click", function (e) {
    e.preventDefault();

    // Add active class for visual feedback
    this.classList.add("back-to-top-active");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });

    // Remove focus and active class after animation completes
    setTimeout(() => {
      this.blur();
      this.classList.remove("back-to-top-active");
    }, 500);
  });

  // Handle touchstart for better mobile response
  backToTopButton.addEventListener(
    "touchstart",
    function (e) {
      this.classList.add("back-to-top-active");
    },
    { passive: true }
  );

  // Handle touchend for mobile devices
  backToTopButton.addEventListener("touchend", function (e) {
    e.preventDefault();

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });

    // Reset the button state after scroll animation
    setTimeout(() => {
      this.blur();
      this.classList.remove("back-to-top-active");
    }, 500);
  });

  // Ensure the button resets if touch is canceled
  backToTopButton.addEventListener("touchcancel", function () {
    this.classList.remove("back-to-top-active");
  });

  // Smooth scrolling for all anchor links
  document
    .querySelectorAll('a[href^="#"]:not(#back-to-top)')
    .forEach((anchor) => {
      anchor.addEventListener("click", function (e) {
        const href = this.getAttribute("href");

        // Only apply smooth scroll for page anchor links
        if (href !== "#") {
          e.preventDefault();

          const targetElement = document.querySelector(href);
          if (targetElement) {
            targetElement.scrollIntoView({
              behavior: "smooth",
            });

            // Update URL hash without jumping
            history.pushState(null, null, href);
          }
        }
      });
    });

  // Collapsible section for architecture figure
  const collapsible = document.querySelector(".collapsible");
  const content = document.querySelector(".content");

  if (collapsible && content) {
    collapsible.addEventListener("click", function () {
      const expanded = this.getAttribute("aria-expanded") === "true";

      this.setAttribute("aria-expanded", !expanded);
      content.setAttribute("aria-hidden", expanded);

      if (!expanded) {
        // Open the collapsible
        content.style.maxHeight = content.scrollHeight + "px";
      } else {
        // Close the collapsible
        content.style.maxHeight = "0";
      }
    });

    // Handle keyboard accessibility
    collapsible.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        collapsible.click();
      }
    });
  }
});
