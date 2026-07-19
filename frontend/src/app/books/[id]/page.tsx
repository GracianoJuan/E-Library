"use client";

import dynamic from "next/dynamic";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import type { Book } from "@/hooks/useBooks";
import { getBook, getBookLikeStatus, getBookRecommendations, likeBook, unlikeBook, readBook } from "@/services/BookService";

const BooksCarousel = dynamic(() => import("@/components/Books.Carousel"), {
	ssr: false,
	loading: () => <div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">Loading recommendations...</div>,
});

export default function BookDetailPage() {
	const params = useParams();
	const router = useRouter();
	const { isLoggedIn, isLoading: authLoading } = useAuth();
	const bookId = Number(Array.isArray(params.id) ? params.id[0] : params.id);
	const [book, setBook] = useState<Book | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [isLiked, setIsLiked] = useState(false);
	const [likeStatusLoading, setLikeStatusLoading] = useState(true);
	const [actionLoading, setActionLoading] = useState(false);
	const [recommendedBooks, setRecommendedBooks] = useState<Book[]>([]);
	const [recommendationsLoading, setRecommendationsLoading] = useState(false);
	const [recommendationsLoaded, setRecommendationsLoaded] = useState(false);
	const recommendationsRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		const loadBook = async () => {
			setLoading(true);
			setError(null);
			try {
				const data = await getBook(bookId);
				setBook(data);
			} catch (err) {
				setError(err instanceof Error ? err.message : "Failed to load book");
			} finally {
				setLoading(false);
			}
		};

		if (!Number.isNaN(bookId)) {
			loadBook();
		}
	}, [bookId]);

	useEffect(() => {
		let active = true;

		const loadLikeStatus = async () => {
			if (authLoading || Number.isNaN(bookId)) {
				return;
			}

			if (!isLoggedIn) {
				setIsLiked(false);
				setLikeStatusLoading(false);
				return;
			}

			setLikeStatusLoading(true);
			try {
				const status = await getBookLikeStatus(bookId);
				if (active) {
					setIsLiked(status.is_liked);
				}
			} catch {
				if (active) {
					setIsLiked(false);
				}
			} finally {
				if (active) {
					setLikeStatusLoading(false);
				}
			}
		};

		loadLikeStatus();

		return () => {
			active = false;
		};
	}, [bookId, isLoggedIn, authLoading]);

	useEffect(() => {
		if (!book || recommendationsLoaded) {
			return;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				const [entry] = entries;
				if (!entry?.isIntersecting) {
					return;
				}

				setRecommendationsLoaded(true);
				setRecommendationsLoading(true);
				getBookRecommendations(book.id, { limit: 10, threshold: 0.25 })
					.then((data) => {
						setRecommendedBooks(data);
					})
					.catch(() => {
						setRecommendedBooks([]);
					})
					.finally(() => {
						setRecommendationsLoading(false);
					});
			},
			{ root: null, threshold: 0.15 }
		);

		if (recommendationsRef.current) {
			observer.observe(recommendationsRef.current);
		}
		return () => observer.disconnect();
	}, [book, recommendationsLoaded]);

	const handleRead = async () => {
		if (!book) return;
		setActionLoading(true);
		try {
			await readBook(book.id);
			router.push(`/books/${book.id}/read`);
		} finally {
			setActionLoading(false);
		}
	};

	const handleLike = async () => {
		if (!book) return;
		if (!isLoggedIn || likeStatusLoading) return;

		const nextLiked = !isLiked;
		setIsLiked(nextLiked);
		setBook((current) =>
			current
				? {
						...current,
						total_likes: current.total_likes + (nextLiked ? 1 : -1),
				  }
				: current
		);

		try {
			if (nextLiked) {
				await likeBook(book.id);
			} else {
				await unlikeBook(book.id);
			}
		} catch {
			// Roll back optimistic update on failure
			setIsLiked(!nextLiked);
			setBook((current) =>
				current
					? {
						...current,
						total_likes: current.total_likes + (nextLiked ? -1 : 1),
					}
					: current
			);
		}
	};

	if (loading) {
		return <div className="py-16 text-center text-sm text-slate-500">Loading book...</div>;
	}

	if (error || !book) {
		return <div className="py-16 text-center text-sm text-red-500">{error ?? "Book not found"}</div>;
	}

	return (
		<div className="grid gap-8 grid-cols-1 md:grid-cols-[240px_minmax(0,1fr)] lg:grid-cols-[320px_minmax(0,1fr)]">
			<div>
				<div className="overflow-hidden rounded-4xl border border-slate-200 bg-white shadow-2xl shadow-slate-950/10 dark:border-slate-800 dark:bg-slate-950">
					<div className="relative aspect-3/4 w-full bg-slate-100 dark:bg-slate-900">
						<Image
							src={book.image_url}
							alt={book.title}
							fill
							className="object-cover"
						/>
					</div>
					<div className="space-y-3 p-5">
						<button
							type="button"
							onClick={handleLike}
							disabled={!isLoggedIn || likeStatusLoading}
							title={isLoggedIn ? "Like this book" : "Login to like books"}
							className="w-full rounded-2xl border border-slate-300 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:border-slate-950 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:text-white dark:hover:bg-slate-900"
						>
							{likeStatusLoading && isLoggedIn ? "Checking..." : isLiked ? "Unlike" : "Like"} ({book.total_likes})
						</button>
						<button
							type="button"
							onClick={handleRead}
							disabled={!isLoggedIn || actionLoading}
							title={isLoggedIn ? "Start reading" : "Login to read this book"}
							className="w-full rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-slate-950 dark:hover:bg-slate-200"
						>
							{actionLoading ? "Opening..." : "Read"}
						</button>
					</div>
				</div>
			</div>
            

			<div className="space-y-6">
				<div>
					<p className="text-sm uppercase tracking-[0.3em] text-slate-500">Book Detail</p>
					<h1 className="mt-3 text-3xl font-black tracking-tight text-slate-950 sm:text-4xl dark:text-white">{book.title}</h1>
					<p className="mt-2 text-base text-slate-500">By {book.author}</p>
				</div>

				<div className="grid gap-4 rounded-4xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-950 sm:grid-cols-2">
					<div>
						<p className="text-xs uppercase tracking-[0.2em] text-slate-500">ISBN</p>
						<p className="mt-2 font-medium text-slate-950 dark:text-white">{book.isbn}</p>
					</div>
					<div>
							<p className="text-xs uppercase tracking-[0.2em] text-slate-500">Genres</p>
							<p className="mt-2 font-medium text-slate-950 dark:text-white">{book.genres?.length ? book.genres.join(", ") : book.category}</p>
					</div>
					<div>
						<p className="text-xs uppercase tracking-[0.2em] text-slate-500">Publisher</p>
						<p className="mt-2 font-medium text-slate-950 dark:text-white">{book.publisher}</p>
					</div>
					<div>
						<p className="text-xs uppercase tracking-[0.2em] text-slate-500">Readers</p>
						<p className="mt-2 font-medium text-slate-950 dark:text-white">{book.total_readers.toLocaleString()}</p>
					</div>
				</div>

				<div className="rounded-4xl border border-slate-200 bg-white p-6 dark:border-slate-800 dark:bg-slate-950">
					<h2 className="text-2xl font-bold text-slate-950 dark:text-white">About this book</h2>
					<p className="mt-4 leading-8 text-slate-600 dark:text-slate-300">
						{book.description}
					</p>
				</div>
			</div>

			<div ref={recommendationsRef} className="mt-14 space-y-6 lg:col-span-2">
				<div>
					<p className="text-sm uppercase tracking-[0.3em] text-slate-500">Recommended</p>
					<h2 className="mt-3 text-3xl font-black tracking-tight text-slate-950 dark:text-white">More like this</h2>
					<p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
						A small set of similar books is loaded only when this section comes into view.
					</p>
				</div>

				{recommendationsLoading ? (
					<div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
						Loading recommendations...
					</div>
				) : recommendedBooks.length > 0 ? (
					<BooksCarousel books={recommendedBooks} />
				) : recommendationsLoaded ? (
					<div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
						No similar books found yet.
					</div>
				) : null}
			</div>
		</div>
	);
}
