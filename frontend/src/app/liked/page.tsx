"use client";

import Link from "next/link";
import Image from "next/image";
import { useEffect, useState } from "react";
import { RequireAuth } from "@/components/AuthGuard";
import { deleteLikedBook, getLikedBooks, type LikedItem } from "@/services/LikedService";

function LikedContent() {
	const [liked, setLiked] = useState<LikedItem[]>([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [processingBookId, setProcessingBookId] = useState<number | null>(null);

	useEffect(() => {
		const loadLiked = async () => {
			setLoading(true);
			setError(null);
			try {
				const data = await getLikedBooks();
				setLiked(data);
			} catch (err) {
				setError(err instanceof Error ? err.message : "Failed to load liked books");
			} finally {
				setLoading(false);
			}
		};

		loadLiked();
	}, []);

	const handleUnlike = async (bookId: number) => {
		setProcessingBookId(bookId);
		setError(null);
		try {
			await deleteLikedBook(bookId);
			setLiked((current) => current.filter((item) => item.book.id !== bookId));
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to unlike book");
		} finally {
			setProcessingBookId(null);
		}
	};

	if (loading) {
		return <div className="py-16 text-center text-sm text-slate-500">Loading liked books...</div>;
	}

	if (error) {
		return <div className="py-16 text-center text-sm text-red-500">{error}</div>;
	}

	return (
		<div className="space-y-6">
			<div>
				<p className="text-sm uppercase tracking-[0.3em] text-slate-500">Liked</p>
				<h1 className="mt-3 text-3xl font-black tracking-tight text-slate-950 sm:text-4xl dark:text-white">Liked Books</h1>
			</div>

			<div className="grid gap-4 sm:grid-cols-4 xl:grid-cols-5">
				{liked.map((item) => (
					<div
						key={item.id}
						className="group overflow-hidden rounded-4xl border border-slate-200 bg-white shadow-lg shadow-slate-950/5 transition hover:-translate-y-1 hover:shadow-2xl dark:border-slate-800 dark:bg-slate-950"
					>
						<Link href={`/books/${item.book.id}`}>
							<div className="relative aspect-3/4 w-full bg-slate-100 dark:bg-slate-900">
								<Image src={item.book.image_url} alt={item.book.title} fill className="object-cover" />
							</div>
							<div className="space-y-2 p-5">
								<p className="text-xs uppercase tracking-[0.2em] text-slate-500">Liked at</p>
								<p className="text-sm text-slate-600 dark:text-slate-300">{new Date(item.created_at).toLocaleString()}</p>
								<h2 className="text-xl font-bold text-slate-950 dark:text-white">{item.book.title}</h2>
								<p className="text-sm text-slate-500">{item.book.author}</p>
							</div>
						</Link>
						<div className="px-5 pb-5">
							<button
								type="button"
								onClick={() => handleUnlike(item.book.id)}
								disabled={processingBookId === item.book.id}
								className="w-full rounded-2xl border border-rose-200 px-4 py-2.5 text-sm font-semibold text-rose-600 transition hover:border-rose-400 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-rose-900/60 dark:text-rose-300 dark:hover:bg-rose-950/40"
							>
								{processingBookId === item.book.id ? "Processing..." : "Unlike"}
							</button>
						</div>
					</div>
				))}
			</div>

			{liked.length === 0 ? (
				<div className="rounded-3xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
					No liked books yet.
				</div>
			) : null}
		</div>
	);
}

export default function LikedPage() {
	return (
		<RequireAuth>
			<LikedContent />
		</RequireAuth>
	);
}