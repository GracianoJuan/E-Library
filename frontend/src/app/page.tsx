"use client";

import { useEffect, useMemo, useState } from "react";
import BookContainer from "../components/BookContainer";
import { useAuth } from "@/hooks/useAuth";
import { getMostLikedBooks, getMostReadBooks } from "@/services/BookService";
import { getRecommendations } from "@/services/RecommendationService";
import { getToken } from "@/services/AuthService";
import type { Book } from "@/hooks/useBooks";

export default function Home() {
  const { isLoggedIn } = useAuth();
  const [recommendedBooks, setRecommendedBooks] = useState<Book[]>([]);
  const [mostLikedBooks, setMostLikedBooks] = useState<Book[]>([]);
  const [mostReadBooks, setMostReadBooks] = useState<Book[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadBooks = async () => {
      setIsLoading(true);
      try {
        const mostLiked = await getMostLikedBooks(15);
        setMostLikedBooks(mostLiked);

        const mostRead = await getMostReadBooks(15);
        setMostReadBooks(mostRead);

        const token = getToken();
        if (isLoggedIn && token) {
          try {
            const recommended = await getRecommendations(15);
            setRecommendedBooks(recommended);
          } catch {
            setRecommendedBooks([]);
          }
        } else {
          setRecommendedBooks([]);
        }
      } finally {
        setIsLoading(false);
      }
    };

    loadBooks();
  }, [isLoggedIn]);

  const heroSubtitle = useMemo(
    () =>
      isLoggedIn
        ? "Continue from where you left off and discover books chosen for you."
        : "Discover books, track your reading, and unlock personal recommendations.",
    [isLoggedIn]
  );

  return (
    <div className="min-h-screen">
      <section className="mb-10 rounded-4xl border border-slate-200 bg-[radial-gradient(circle_at_top,rgba(15,23,42,0.95),rgba(15,23,42,0.86)_45%,rgba(2,6,23,1)_100%)] px-5 py-12 text-white shadow-2xl shadow-slate-950/20 sm:px-8 sm:py-16 md:mb-12 md:py-24">
        <div className="mx-auto max-w-4xl text-center">
          <p className="mb-4 text-xs uppercase tracking-[0.35em] text-white/50 sm:text-sm">E-Library</p>
          <h1 className="text-3xl font-black tracking-tight sm:text-5xl md:text-7xl">
            Find your next book obsession.
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-sm leading-7 text-white/75 sm:text-lg md:text-xl">
            {heroSubtitle}
          </p>
        </div>
      </section>

      <div className="space-y-12">
        {isLoggedIn && recommendedBooks.length > 0 && (
          <BookContainer title="Recommended for You" books={recommendedBooks} />
        )}
        {isLoading ? (
          <div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
            Loading books...
          </div>
        ) : null}
        <BookContainer title="Most Liked" books={mostLikedBooks} />
        <BookContainer title="Most Read" books={mostReadBooks} />
      </div>
    </div>
  );
}