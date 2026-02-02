/**
 * ============================================================
 * COSMETIKA — GSAP Animation Controller
 * ============================================================
 * Scroll-triggered reveals, parallax, header hide/show,
 * staggered product grid, and hero entrance.
 *
 * TOMORROW (Phase 1):
 * After extracting motion values from the reference site
 * using Window MCP, update the CONFIG object below with
 * the exact durations, easings, and offsets.
 * ============================================================
 */

(function () {
  'use strict';

  // ---- CONFIG (update with reference site values) ----
  const CONFIG = {
    // Durations (seconds)
    fadeUpDuration: 1,
    heroZoomDuration: 1.5,
    staggerInterval: 0.1,
    headerHideThreshold: 80,

    // Easing — GSAP easing strings
    // Reference: https://gsap.com/docs/v3/Eases/
    defaultEase: 'power3.out',
    heroEase: 'power2.out',
    staggerEase: 'power2.out',

    // ScrollTrigger defaults
    triggerStart: 'top 85%',
    triggerEnd: 'bottom 20%',
  };

  // ---- FADE ANIMATIONS (fade-up, fade-in, fade-left, fade-right, scale-in) ----
  function initFadeAnimations() {
    const animTypes = {
      'fade-up':    { y: 40, opacity: 0 },
      'fade-in':    { opacity: 0 },
      'fade-left':  { x: -40, opacity: 0 },
      'fade-right': { x: 40, opacity: 0 },
      'scale-in':   { scale: 0.92, opacity: 0 },
    };

    Object.entries(animTypes).forEach(([type, fromVars]) => {
      const elements = document.querySelectorAll(`[data-animate="${type}"]`);
      elements.forEach((el) => {
        const delay = parseFloat(el.dataset.delay) || 0;

        gsap.fromTo(el, fromVars, {
          y: 0,
          x: 0,
          scale: 1,
          opacity: 1,
          duration: CONFIG.fadeUpDuration,
          delay: delay,
          ease: CONFIG.defaultEase,
          scrollTrigger: {
            trigger: el,
            start: CONFIG.triggerStart,
            once: true,
          },
          onComplete: () => el.classList.add('is-animated'),
        });
      });
    });
  }

  // ---- HERO ----
  function initHeroAnimation() {
    const hero = document.querySelector('[data-animate="hero"]');
    if (!hero) return;

    const media = hero.querySelector('.cosmetika-hero__image, .cosmetika-hero__video');
    if (media) {
      gsap.fromTo(media,
        { scale: 1.15 },
        {
          scale: 1,
          duration: CONFIG.heroZoomDuration,
          ease: CONFIG.heroEase,
          onComplete: () => hero.classList.add('is-animated'),
        }
      );
    }

    // Parallax on hero media during scroll
    if (media) {
      gsap.to(media, {
        yPercent: 20,
        ease: 'none',
        scrollTrigger: {
          trigger: hero,
          start: 'top top',
          end: 'bottom top',
          scrub: true,
        },
      });
    }
  }

  // ---- STAGGER GRID (product cards) ----
  function initStaggerGrids() {
    const grids = document.querySelectorAll('[data-animate="stagger-grid"]');
    grids.forEach((grid) => {
      const cards = grid.querySelectorAll('.cosmetika-product-card, [data-animate="fade-up"]');
      if (!cards.length) return;

      gsap.fromTo(cards,
        { y: 30, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: CONFIG.fadeUpDuration,
          ease: CONFIG.staggerEase,
          stagger: CONFIG.staggerInterval,
          scrollTrigger: {
            trigger: grid,
            start: CONFIG.triggerStart,
            once: true,
          },
          onComplete: () => grid.classList.add('is-animated'),
        }
      );
    });
  }

  // ---- PARALLAX ----
  function initParallax() {
    const parallaxEls = document.querySelectorAll('[data-parallax]');
    parallaxEls.forEach((el) => {
      const speed = parseFloat(el.dataset.parallax) || 0.2;
      gsap.to(el, {
        yPercent: speed * 100,
        ease: 'none',
        scrollTrigger: {
          trigger: el,
          start: 'top bottom',
          end: 'bottom top',
          scrub: true,
        },
      });
    });
  }

  // ---- HEADER HIDE/SHOW ON SCROLL ----
  function initHeaderScroll() {
    const header = document.querySelector('.cosmetika-header');
    if (!header) return;

    let lastScroll = 0;

    ScrollTrigger.create({
      start: 'top top',
      end: 'max',
      onUpdate: (self) => {
        const scrollY = self.scroll();
        const direction = self.direction; // 1 = down, -1 = up

        if (scrollY > CONFIG.headerHideThreshold) {
          header.classList.add('cosmetika-header--scrolled');
        } else {
          header.classList.remove('cosmetika-header--scrolled');
        }

        if (direction === 1 && scrollY > 300) {
          header.classList.add('cosmetika-header--hidden');
        } else {
          header.classList.remove('cosmetika-header--hidden');
        }

        lastScroll = scrollY;
      },
    });
  }

  // ---- IMAGE REVEALS ----
  function initImageReveals() {
    const reveals = document.querySelectorAll('.cosmetika-img-reveal');
    reveals.forEach((container) => {
      const img = container.querySelector('img');
      if (!img) return;

      if (img.complete) {
        container.classList.add('is-loaded');
      } else {
        img.addEventListener('load', () => container.classList.add('is-loaded'), { once: true });
      }
    });
  }

  // ---- CAROUSEL (prev/next arrows + drag scroll) ----
  function initCarousels() {
    document.querySelectorAll('[data-carousel-track]').forEach(function(track) {
      var section = track.closest('.cosmetika-carousel');
      if (!section) return;

      var prevBtn = section.querySelector('[data-carousel-prev]');
      var nextBtn = section.querySelector('[data-carousel-next]');
      var slideWidth = function() {
        var slide = track.querySelector('.cosmetika-carousel__slide');
        if (!slide) return 300;
        var style = getComputedStyle(track);
        var gap = parseFloat(style.gap) || 0;
        return slide.offsetWidth + gap;
      };

      if (nextBtn) {
        nextBtn.addEventListener('click', function() {
          track.scrollBy({ left: slideWidth(), behavior: 'smooth' });
        });
      }
      if (prevBtn) {
        prevBtn.addEventListener('click', function() {
          track.scrollBy({ left: -slideWidth(), behavior: 'smooth' });
        });
      }

      // Touch / drag scroll
      var isDown = false, startX, scrollLeft;
      track.addEventListener('mousedown', function(e) {
        isDown = true; startX = e.pageX - track.offsetLeft; scrollLeft = track.scrollLeft;
        track.style.cursor = 'grabbing';
      });
      track.addEventListener('mouseleave', function() { isDown = false; track.style.cursor = ''; });
      track.addEventListener('mouseup', function() { isDown = false; track.style.cursor = ''; });
      track.addEventListener('mousemove', function(e) {
        if (!isDown) return;
        e.preventDefault();
        var x = e.pageX - track.offsetLeft;
        track.scrollLeft = scrollLeft - (x - startX);
      });
    });
  }

  // ---- POPUP ----
  function initPopup() {
    // Popup logic is inline in the section for portability
  }

  // ---- BOOT ----
  function init() {
    if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
      // Still init non-GSAP features
      initCarousels();
      return;
    }

    gsap.registerPlugin(ScrollTrigger);

    initFadeAnimations();
    initHeroAnimation();
    initStaggerGrids();
    initParallax();
    initHeaderScroll();
    initImageReveals();
    initCarousels();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
