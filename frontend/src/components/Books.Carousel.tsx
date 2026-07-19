"use client";

import Image from "next/image";
import Link from "next/link";
import { useRef } from "react";

interface Book {
  id: string | number;
  title: string;
  image_url?: string;
  image?: string;
}

interface BooksCarouselProps {
  /**
   * Array of books to display in the carousel. Each book must include an `id`
   * which corresponds to the detail page (`/books/[id]`), a `title` and an
   * `image` URL that can be consumed by `next/image`.
   */
  books: Book[];
}

export default function BooksCarousel({ books }: BooksCarouselProps) {
  const carouselRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: "left" | "right") => {
    if (!carouselRef.current) return;
    const width = carouselRef.current.offsetWidth;
    carouselRef.current.scrollBy({
      left: direction === "left" ? -width : width,
      behavior: "smooth",
    });
  };

  return (
    <div className="relative group">
      {/* navigation buttons */}
      <button
        onClick={() => scroll("left")}
        className="absolute -left-6 top-1/2 z-10 hidden h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-blue-600 text-lg font-bold text-white opacity-0 duration-200 hover:scale-110 hover:bg-blue-700 hover:shadow-xl group-hover:opacity-100 md:flex md:-left-8"
      >
        &#8249;
      </button>
      <button
        onClick={() => scroll("right")}
        className="absolute -right-6 top-1/2 z-10 hidden h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-blue-600 text-lg font-bold text-white opacity-0 transition-all duration-200 hover:scale-110 hover:bg-blue-700 hover:shadow-xl group-hover:opacity-100 md:flex md:-right-8"
      >
        &#8250;
      </button>

      {/* scrollable area */}
      <div
        ref={carouselRef}
        className="flex snap-x snap-mandatory gap-4 overflow-x-auto px-1 pb-2 scroll-smooth sm:gap-6 sm:px-2"
      >
        {books.map((book) => (
            <Link
              key={book.id}
              href={`/books/${book.id}`}
              className="group relative h-72 shrink-0 snap-start overflow-hidden rounded-2xl bg-white shadow-lg transition-all duration-300 hover:-translate-y-3 hover:shadow-2xl w-[72vw] sm:w-44 md:w-48 lg:w-56"
            >
            <div className="relative w-full h-full">
              <Image
                src={book.image_url ?? book.image ?? "/placeholder-book.png"}
                alt={book.title}
                fill
                sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 25vw"
                className="object-cover group-hover:scale-110 transition-transform duration-300"
              />
              <div className="absolute inset-0 flex items-end bg-black/60 p-4 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
                <p className="text-white font-semibold text-sm line-clamp-3">{book.title}</p>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}