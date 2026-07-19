"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getBookContentInfo, getBookContentPageUrl } from "@/services/BookService";

interface BookImageReaderProps {
	bookId: number;
}

type BookContentInfo = {
	page_numbers: number[];
};

function getPageWindow(currentIndex: number, totalPages: number): number[] {
	if (totalPages <= 0) {
		return [];
	}

	const pages: number[] = [];
	const start = Math.max(0, currentIndex - 1);
	const end = Math.min(totalPages - 1, currentIndex + 1);

	for (let index = start; index <= end; index += 1) {
		pages.push(index);
	}

	if (pages.length < 3 && start === 0) {
		const needed = 3 - pages.length;
		for (let index = end + 1; index <= Math.min(totalPages - 1, end + needed); index += 1) {
			pages.push(index);
		}
	}

	if (pages.length < 3 && end === totalPages - 1) {
		const needed = 3 - pages.length;
		for (let index = start - 1; index >= Math.max(0, start - needed); index -= 1) {
			pages.unshift(index);
		}
	}

	return pages;
}

function BookPageImage({
	bookId,
	pageIndex,
	pageNumber,
	shouldRender,
	onVisible,
	scale,
}: {
	bookId: number;
	pageIndex: number;
	pageNumber: number;
	shouldRender: boolean;
	onVisible: (pageIndex: number) => void;
	scale: number;
}) {
	const containerRef = useRef<HTMLDivElement | null>(null);
	const [loaded, setLoaded] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		if (!containerRef.current) {
			return;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				const [entry] = entries;
				if (entry?.isIntersecting) {
					onVisible(pageIndex);
				}
			},
			{
				root: null,
				threshold: 0.1,
			}
		);

		observer.observe(containerRef.current);
		return () => observer.disconnect();
	}, [onVisible, pageIndex]);

	useEffect(() => {
		if (shouldRender) {
			setLoaded(false);
			setError(null);
		}
	}, [pageNumber, shouldRender]);

	const imageUrl = getBookContentPageUrl(bookId, pageNumber);
	const zoomWidth = `${Math.round(scale * 100)}%`;

	return (
		<div ref={containerRef} data-page={pageNumber} className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950">
			<div className="mb-3 text-right text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Page {pageNumber}</div>
			<div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 dark:border-slate-800 dark:bg-slate-900/40">
				{shouldRender ? (
					<div className="relative">
						{!loaded && <div className="h-175 animate-pulse bg-slate-200/60 dark:bg-slate-800/60" />}
						{error ? (
							<div className="py-20 text-center text-sm text-red-500">{error}</div>
						) : (
							<img
								src={imageUrl}
								alt={`Page ${pageNumber}`}
								loading="lazy"
								onLoad={() => setLoaded(true)}
								onError={() => setError("Failed to load page image")}
								className={`mx-auto block h-auto max-w-none transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
								style={{ width: zoomWidth }}
							/>
						)}
					</div>
				) : (
					<div className="h-175 bg-slate-200/40 dark:bg-slate-800/30" />
				)}
			</div>
		</div>
	);
}

export default function BookImageReader({ bookId }: BookImageReaderProps) {
	const [pageNumbers, setPageNumbers] = useState<number[]>([]);
	const [currentIndex, setCurrentIndex] = useState(0);
	const [scale, setScale] = useState(1.1);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);

	useEffect(() => {
		let mounted = true;

		const load = async () => {
			try {
				setLoading(true);
				setError(null);
				const data = (await getBookContentInfo(bookId)) as BookContentInfo;
				if (mounted) {
					setPageNumbers(data.page_numbers ?? []);
					setCurrentIndex(0);
				}
			} catch (loadError) {
				if (mounted) {
					setError(loadError instanceof Error ? loadError.message : "Failed to load book pages");
				}
			} finally {
				if (mounted) {
					setLoading(false);
				}
			}
		};

		load();

		return () => {
			mounted = false;
		};
	}, [bookId]);

	const activePages = useMemo(() => getPageWindow(currentIndex, pageNumbers.length), [currentIndex, pageNumbers.length]);

	if (loading) {
		return <div className="py-16 text-center text-sm text-slate-500">Loading pages...</div>;
	}

	if (error) {
		return <div className="py-16 text-center text-sm text-red-500">{error}</div>;
	}

	if (pageNumbers.length === 0) {
		return <div className="py-16 text-center text-sm text-slate-500">No page images found for this book.</div>;
	}

	return (
		<div>
			<div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 dark:border-slate-800 dark:bg-slate-950">
				<p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
					Page {pageNumbers[currentIndex] ?? 1} / {pageNumbers.length}
				</p>
				<div className="flex items-center gap-2">
					<button
						type="button"
						onClick={() => setScale((value) => Math.max(0.9, value - 0.15))}
						className="rounded-full border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-slate-950 hover:text-slate-950 dark:border-slate-700 dark:text-slate-300 dark:hover:border-white dark:hover:text-white"
					>
						Zoom Out
					</button>
					<span className="min-w-12 text-center text-xs font-semibold text-slate-600 dark:text-slate-300">{Math.round(scale * 100)}%</span>
					<button
						type="button"
						onClick={() => setScale((value) => Math.min(2.2, value + 0.15))}
						className="rounded-full border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-slate-950 hover:text-slate-950 dark:border-slate-700 dark:text-slate-300 dark:hover:border-white dark:hover:text-white"
					>
						Zoom In
					</button>
				</div>
			</div>

			<div className="space-y-6">
				{pageNumbers.map((pageNumber, index) => (
					<BookPageImage
						key={`${pageNumber}-${scale}`}
						bookId={bookId}
						pageIndex={index}
						pageNumber={pageNumber}
						shouldRender={activePages.includes(index)}
						scale={scale}
						onVisible={setCurrentIndex}
					/>
				))}
			</div>
		</div>
	);
}