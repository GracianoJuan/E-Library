import BooksCarousel from "./Books.Carousel";
import type { Book } from "@/hooks/useBooks";

interface BookContainerProps {
  title: string;
  books: Book[];
}

export default function BookContainer({ title, books }: BookContainerProps) {
  return (
    <section className="border-b border-gray-200 py-10 last:border-b-0 dark:border-gray-800 md:py-14">
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:gap-3">
        <div className="h-6 w-12 rounded bg-black sm:h-8 sm:w-1"></div>
        <h2 className="text-2xl font-bold text-slate-950 sm:text-3xl md:text-4xl dark:text-white">
          {title}
        </h2>
      </div>
      <BooksCarousel books={books} />
    </section>
  );
};
